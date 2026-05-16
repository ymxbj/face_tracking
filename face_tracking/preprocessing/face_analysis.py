# coding: utf-8

"""Face detection + 106-pt alignment via the upstream ``insightface`` package."""


import time
from typing import List, Optional, Sequence

import numpy as np


def _sort_by_direction(faces, direction: str = "large-small", face_center=None):
    """Sort face detections by a chosen heuristic. Mirrors LivePortrait."""
    if not faces:
        return faces

    if direction == "left-right":
        return sorted(faces, key=lambda f: f["bbox"][0])
    if direction == "right-left":
        return sorted(faces, key=lambda f: f["bbox"][0], reverse=True)
    if direction == "top-bottom":
        return sorted(faces, key=lambda f: f["bbox"][1])
    if direction == "bottom-top":
        return sorted(faces, key=lambda f: f["bbox"][1], reverse=True)
    if direction == "small-large":
        return sorted(
            faces,
            key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
        )
    if direction == "large-small":
        return sorted(
            faces,
            key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
            reverse=True,
        )
    if direction == "distance-from-retarget-face" and face_center is not None:
        return sorted(
            faces,
            key=lambda f: (
                ((f["bbox"][2] + f["bbox"][0]) / 2 - face_center[0]) ** 2
                + ((f["bbox"][3] + f["bbox"][1]) / 2 - face_center[1]) ** 2
            ) ** 0.5,
        )
    return faces


class FaceAnalysisDIY:
    """Thin wrapper around ``insightface.app.FaceAnalysis``.

    The upstream class loads several ONNX models (RetinaFace detector,
    ArcFace recogniser, 2d-106 landmark, 3d-68 landmark, gender/age head)
    from the ``buffalo_l`` model pack. We expose a simplified ``get()``
    method that runs detection + the 106-pt landmark head only.
    """

    def __init__(
        self,
        name: str = "buffalo_l",
        root: str = "~/.insightface",
        providers: Optional[Sequence[str]] = None,
    ):
        from insightface.app import FaceAnalysis

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self._app = FaceAnalysis(
            name=name, root=root, providers=list(providers),
        )

    def prepare(self, ctx_id: int = 0, det_size=(512, 512), det_thresh: float = 0.1):
        self._app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=det_thresh)
        return self

    def get(
        self,
        img_bgr: np.ndarray,
        max_face_num: int = 0,
        flag_do_landmark_2d_106: bool = True,
        direction: str = "large-small",
        face_center=None,
    ) -> List:
        from insightface.app.common import Face

        det_model = self._app.models.get("detection") or self._app.det_model
        bboxes, kpss = det_model.detect(img_bgr, max_num=max_face_num, metric="default")
        if bboxes.shape[0] == 0:
            return []

        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = kpss[i] if kpss is not None else None
            face = Face(bbox=bbox, kps=kps, det_score=det_score)
            for taskname, model in self._app.models.items():
                if taskname == "detection":
                    continue
                if (not flag_do_landmark_2d_106) and taskname == "landmark_2d_106":
                    continue
                model.get(img_bgr, face)
            ret.append(face)

        return _sort_by_direction(ret, direction, face_center)

    def warmup(self) -> float:
        t0 = time.perf_counter()
        self.get(np.zeros((512, 512, 3), dtype=np.uint8))
        return time.perf_counter() - t0

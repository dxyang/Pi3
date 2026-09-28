import sys
sys.path.append('.')

import os.path as osp
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from datasets.base.base_dataset import BaseDataset

DEFAULT_MANIFEST = osp.join(osp.dirname(__file__), 'divercam_manifest.yaml')


class DiverCamDataset(BaseDataset):
    """ZED X Mini reef surveys supervised by Metashape/AprilTag pseudo-GT.

    One class covers every dive; `divercam_manifest.yaml` says which dives exist,
    where their COLMAP index lives, and whether their reconstruction is metrically
    scaled. Adding a newly-processed dive is one manifest entry.

    Supervision is the sparse SfM point cloud (~5.5k observations/frame), carried
    as (u, v, Z_cam) triples and rasterized at the *output* resolution. It cannot
    go through the depthmap resize path: that uses INTER_NEAREST, which at ~0.2%
    density would discard ~98% of the points without raising anything.

    Each dive reports its own `dataset` label so the metric-scale loss can be
    gated per dive -- some exports are AprilTag-scaled, others are not.
    """

    def __init__(
        self,
        manifest=DEFAULT_MANIFEST,
        dives=None,             # subset of manifest labels; None = all
        split='train',          # 'train' | 'val' | 'all'
        val_frac=0.15,          # tail fraction of each segment held out
        frame_stride=1,         # int, or a list sampled per window (overlap knob)
        metric_only=False,      # drop dives whose reconstruction is unscaled
        use_gt_audit=True,      # exclude frames the EKF cross-check flagged
        verbose=True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.dataset_label = 'DiverCam'
        self.split = split
        self.val_frac = val_frac

        # A window's temporal span is frame_num * stride, so a larger stride covers
        # more seafloor with the same number of frames -- the low-overlap regime.
        # Sampling it per window trains one model across framerates instead of one
        # per rate, and doubles as baseline augmentation.
        self.strides = ([int(frame_stride)] if isinstance(frame_stride, (int, np.integer))
                        else [int(s) for s in frame_stride])
        self.frame_stride = min(self.strides)

        with open(manifest) as f:
            entries = yaml.safe_load(f)['dives']

        self.dives = []
        for e in entries:
            if dives is not None and e['label'] not in dives:
                continue
            if metric_only and not e.get('metric', False):
                continue
            self.dives.append(e)

        if not self.dives:
            raise ValueError(f'No dives selected from {manifest}')

        # (dive_idx, seg_start, seg_end) for every usable segment in this split
        self.segments = []
        self._index_cache = {}
        for di, dive in enumerate(self.dives):
            meta = self._meta(di)
            bounds = self._bounds_with_audit(di, meta) if use_gt_audit else meta['segment_bounds']
            for a, b in zip(bounds[:-1], bounds[1:]):
                a, b = int(a), int(b)
                n = b - a
                cut = int(round(n * (1.0 - self.val_frac)))
                if split == 'train':
                    a, b = a, a + cut
                elif split == 'val':
                    a, b = a + cut, b
                # a window needs frame_num frames spaced by frame_stride
                if b - a >= self.frame_num * self.frame_stride:
                    self.segments.append((di, a, b))

        if not self.segments:
            raise ValueError(
                f'No segment in split={split} is long enough for frame_num='
                f'{self.frame_num} at stride {self.frame_stride}')

        # Every valid window start, so an epoch covers the data once and a given
        # index always yields the same window. Windows overlap by design -- the
        # stride between them is 1 frame, independent of frame_stride *within* a
        # window, which is the overlap knob for the low-framerate experiments.
        # Carry each window's segment end alongside its start: the sampler varies
        # frame_num per batch (train.image_num_range), so the span is not known
        # here and _get_views has to clamp against the segment it belongs to.
        span = self.frame_num * self.frame_stride
        self.starts = np.array(
            [(di, s, b) for di, a, b in self.segments for s in range(a, max(a + 1, b - span + 1))],
            dtype=np.int64)

        if verbose:
            total = sum(b - a for _, a, b in self.segments)
            print(f'[{self.dataset_label}] {len(self.dives)} dive(s), '
                  f'{len(self.segments)} segment(s), {total} frames, '
                  f'{len(self.starts)} windows in split={split} '
                  f'(frame_num={self.frame_num}, frame_stride={self.frame_stride})')

    # ---------------------------------------------------------------- index io

    def _meta(self, di):
        """Lazily open a dive's index. uvz is memory-mapped -- it is ~0.5GB/dive."""
        if di not in self._index_cache:
            dive = self.dives[di]
            idx = Path(dive['root']) / dive['export'] / 'index'
            if not idx.is_dir():
                raise FileNotFoundError(
                    f"No index at {idx}. Build it with scripts/prepare_divercam_index.py")
            meta = dict(np.load(idx / 'meta.npz', allow_pickle=False))
            meta['uvz'] = np.load(idx / 'uvz.npy', mmap_mode='r')
            meta['offsets'] = np.load(idx / 'offsets.npy')
            # The index stores the export verbatim, in COLMAP's convention (top-left
            # pixel centre at 0.5). Everything downstream -- camera_matrix_of_crop
            # included -- works in OpenCV's (centre at 0). Shift the principal point
            # here and the observations in _get_views, so the two stay consistent.
            meta['intrinsics'] = meta['intrinsics'].copy()
            meta['intrinsics'][:2, 2] -= 0.5
            self._index_cache[di] = meta
        return self._index_cache[di]

    def _bounds_with_audit(self, di, meta):
        """Treat EKF-flagged frames as extra segment boundaries.

        scripts/audit_gt_vs_ekf.py compares Metashape's frame-to-frame motion
        against the DVL's and flags pairs that disagree grossly -- reconstruction
        defects the >10x-median break detector misses because they sit inside a
        segment (e.g. spurious translation while hovering and rotating). Splitting
        there means no training window can span one.
        """
        idx = Path(self.dives[di]['root']) / self.dives[di]['export'] / 'index'
        audit = idx / 'gt_audit.npz'
        if not audit.exists():
            return meta['segment_bounds']
        trust = np.load(audit)['trust']
        cuts = set(int(b) for b in meta['segment_bounds'])
        bad = np.flatnonzero(~trust)
        for i in bad:                      # drop frame i by cutting on both sides
            cuts.add(int(i))
            cuts.add(int(i) + 1)
        bounds = np.array(sorted(c for c in cuts if 0 <= c <= len(trust)), dtype=np.int64)
        # A flagged frame becomes its own 1-frame segment, which no window can use.
        return bounds

    @staticmethod
    def _c2w(qvec, tvec):
        w, x, y, z = qvec
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
            [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
        ])
        # COLMAP stores world-to-camera; Pi3 wants cam-to-world.
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ tvec
        return c2w

    @staticmethod
    def _rasterize(uvz, K_in, K_out, out_wh):
        """Map observations through the crop/resize and splat them onto the output grid.

        Crops and uniform rescales leave the ray through a pixel unchanged, so the
        mapping is recoverable from the two intrinsic matrices alone -- no need to
        track the sequence of operations _crop_resize_if_necessary applied.
        """
        W, H = out_wh
        u = (uvz[:, 0] - K_in[0, 2]) * (K_out[0, 0] / K_in[0, 0]) + K_out[0, 2]
        v = (uvz[:, 1] - K_in[1, 2]) * (K_out[1, 1] / K_in[1, 1]) + K_out[1, 2]

        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        ok = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

        depth = np.full((H, W), np.inf, dtype=np.float32)
        if ok.any():
            # Nearest surface wins where several points land on one pixel.
            np.minimum.at(depth, (vi[ok], ui[ok]), uvz[ok, 2].astype(np.float32))
        depth[~np.isfinite(depth)] = 0.0
        return depth

    # ---------------------------------------------------------------- sampling

    def __len__(self):
        return len(self.starts)

    def _get_views(self, index, resolution, rng):
        di, start, seg_end = (int(v) for v in self.starts[index % len(self.starts)])
        dive = self.dives[di]
        meta = self._meta(di)

        seg_start = next(a for _, a, b in self.segments if a <= start < b)
        seg_len = seg_end - seg_start

        # Only strides whose window actually fits inside this segment.
        fits = [s for s in self.strides if self.frame_num * s <= seg_len]
        stride = int(rng.choice(fits)) if fits else min(self.strides)
        span = self.frame_num * stride

        if self.frame_num > 20 and rng.random() < self.random_sample_thres:
            idxs = sorted(rng.choice(np.arange(seg_start, seg_end), size=self.frame_num,
                                     replace=seg_len < self.frame_num))
        else:
            # Clamp inside the segment: frame_num and stride both vary per window,
            # and a window must never cross a break.
            start = max(min(start, seg_end - span), seg_start)
            idxs = [min(start + k * stride, seg_end - 1) for k in range(self.frame_num)]

        self.this_views_info = dict(scene=dive['label'], pairs=idxs)

        K0 = meta['intrinsics']
        names = meta['names']
        left_dir = Path(dive['root']) / 'left'

        views = []
        for i in idxs:
            name = str(names[i])
            rgb = np.array(Image.open(left_dir / name).convert('RGB'))

            # The depthmap handed to _crop_resize_if_necessary is a placeholder:
            # we only want the image and the updated intrinsics out of it, then
            # rasterize the sparse points ourselves at the output resolution.
            placeholder = np.zeros(rgb.shape[:2], dtype=np.float32)
            rgb, _, K = self._crop_resize_if_necessary(
                rgb, placeholder, K0.copy(), resolution, rng=rng, info=name)

            lo, hi = meta['offsets'][i], meta['offsets'][i + 1]
            uvz = np.asarray(meta['uvz'][lo:hi], dtype=np.float64).copy()
            uvz[:, :2] -= 0.5                      # COLMAP -> OpenCV pixel convention
            depthmap = self._rasterize(uvz, K0, K, rgb.size)

            views.append(dict(
                img=rgb,
                depthmap=depthmap,
                camera_pose=self._c2w(meta['qvec'][i], meta['tvec'][i]),
                camera_intrinsics=K.astype(np.float32),
                dataset=dive['label'],
                label=dive['label'],
                instance=name,
            ))
        return views

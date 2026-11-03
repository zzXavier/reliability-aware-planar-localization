import io
import json
import os
import sys
import zipfile
from pathlib import Path, PurePosixPath

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .geometry import assess, corners, project, resize_iso


class Matchers:
    def __init__(self, upstream, count):
        # XFeat chooses the device in its constructor. Apply before importing torch.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(17)
        cv2.setNumThreads(1)
        cv2.setRNGSeed(17)
        sys.path.insert(0, str(upstream))
        from modules.xfeat import XFeat
        weights = torch.load(upstream/"weights/xfeat.pt", map_location="cpu", weights_only=True)
        self.xfeat = XFeat(weights=weights, top_k=count)
        if self.xfeat.dev.type != "cpu":
            raise RuntimeError("Benchmark is pinned to CPU")
        self.orb = cv2.ORB_create(nfeatures=count)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def extract(self, rgb):
        kp, des = self.orb.detectAndCompute(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), None)
        orb_points = np.array([p.pt for p in kp], np.float32).reshape(-1, 2)
        xf = self.xfeat.detectAndCompute(self.xfeat.parse_input(rgb))[0]
        return {"ORB": (orb_points, des), "XFeat": (xf["keypoints"].cpu().numpy(), xf["descriptors"])}

    def correspondences(self, first, second, method):
        a, fa = first[method]
        b, fb = second[method]
        if not len(a) or not len(b):
            return a[:0], b[:0]
        if method == "ORB":
            matches = self.bf.match(fa, fb)
            ia = [m.queryIdx for m in matches]
            ib = [m.trainIdx for m in matches]
        else:
            ia, ib = self.xfeat.match(fa, fb, min_cossim=-1)
            ia, ib = ia.cpu().numpy(), ib.cpu().numpy()
        return a[ia], b[ib]


def run_pair(matchers, reference, scene, first, second, truth, cfg):
    for method in ("ORB", "XFeat"):
        a, b = matchers.correspondences(first, second, method)
        result = assess(a, b, reference.shape[1::-1], scene.shape[1::-1], truth, cfg["ransac_px"])
        yield method, result


def benchmark(archive, matchers, cfg):
    rows = []
    with zipfile.ZipFile(archive) as z:
        folders = sorted(PurePosixPath(p).parent for p in z.namelist() if p.endswith("/H_1_2") and
                         PurePosixPath(p).parent.name.startswith(("i_", "v_")) and not p.startswith("__MACOSX/"))
        if len(folders) != 116 or len(set(folders)) != len(folders):
            raise ValueError(f"Expected 116 canonical HPatches sequences, found {len(folders)}")
        dev = set()
        for prefix in ("i_", "v_"):
            dev.update([p for p in folders if p.name.startswith(prefix)][::3])
        for ix, folder in enumerate(folders):
            def read(index):
                raw = z.read(str(folder/f"{index}.ppm"))
                arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if arr is None:
                    raise ValueError(f"Cannot decode {folder}/{index}.ppm")
                return resize_iso(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), cfg["max_side"])
            ref, s1 = read(1)
            first = matchers.extract(ref)
            for number in range(2, 7):
                target, sn = read(number)
                h = np.loadtxt(io.BytesIO(z.read(str(folder/f"H_1_{number}"))))
                true = sn @ h @ np.linalg.inv(s1)
                second = matchers.extract(target)
                for method, result in run_pair(matchers, ref, target, first, second, true, cfg):
                    rows.append(dict(sequence=folder.name, target=number, method=method,
                                     kind="illumination" if folder.name.startswith("i_") else "viewpoint",
                                     split="dev" if folder in dev else "test", **result))
            if (ix+1)%10 == 0:
                print(f"{ix+1}/{len(folders)} sequences", flush=True)
    return rows


def demo(matchers, cfg, out):
    # Original generated poster: no external image or display permissions needed.
    rng = np.random.default_rng(17)
    poster = Image.new("RGB", (448,320), "#e8e3ca")
    draw = ImageDraw.Draw(poster)
    for _ in range(160):
        x, y = rng.integers((8,8),(433,306))
        color = tuple(int(v) for v in rng.integers(30,225,size=3))
        draw.rectangle((int(x), int(y),int(x+7),int(y+7)),fill=color)
    for y in range(30, 290, 60):
        draw.text((24,y),"PLANAR LOCALIZATION / REPRODUCIBLE TEST",fill="black")
    a = np.array(poster)
    quad = np.float32([[25,22],[423,10],[400,303],[10,288]])
    h = cv2.getPerspectiveTransform(corners((448,320)).astype(np.float32), quad)
    b = cv2.warpPerspective(a, h, (448,320))
    f1, f2 = matchers.extract(a), matchers.extract(b)
    canvas = Image.new("RGB", (896, 748), "#152536")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17)
    small = ImageFont.load_default(size=13)
    draw.text((10,7),"Planar localization on a known perspective warp",font=font,fill="white")
    draw.text((10,32),"Left: reference. Right: warped target. Lines: 12 example matches. Orange: truth. Blue: estimate.",
              font=small,fill="#dbeaf3")
    measurements = {}
    for index, (method, row) in enumerate(run_pair(matchers,a,b,f1,f2,h,cfg)):
        y = 80 + index*347
        corner_text = "no estimate" if row["corner_error"] is None else f"{row['corner_error']:.2f} px corner error"
        draw.text((10,y-23),f"{method} | {row['matches']} matches | {corner_text}",font=small,fill="white")
        canvas.paste(poster,(0,y))
        image = Image.fromarray(b.copy())
        overlay = ImageDraw.Draw(image)
        overlay.line([tuple(p) for p in quad] + [tuple(quad[0])],fill="#f17022",width=4)
        if row["homography"]:
            pred = project(np.array(row["homography"]), corners((448,320)))
            points = [tuple(p) for p in pred]
            overlay.line(points + [points[0]], fill="#167799",width=2)
        canvas.paste(image,(448,y))
        source, target = matchers.correspondences(f1,f2,method)
        for j in np.linspace(0, max(len(source)-1, 0), min(12,len(source)),dtype=int):
            p, q = source[j], target[j]
            x1, y1, x2, y2 = float(p[0]), float(p[1]+y), float(q[0]+448), float(q[1]+y)
            color = "#00a99a" if np.linalg.norm(project(h,p[None,:])[0]-q) <= 3 else "#f45a69"
            draw.line((x1,y1,x2,y2),fill=color,width=1)
            for u,v in ((x1,y1),(x2,y2)):
                draw.ellipse((u-2,v-2,u+2,v+2), fill=color)
        measurements[method]=row
    out.mkdir(parents=True,exist_ok=True)
    canvas.save(out/"demo.png")
    (out/"demo.json").write_text(json.dumps(measurements, indent=2)+"\n",encoding="utf-8")

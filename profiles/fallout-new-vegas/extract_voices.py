"""Fallout: New Vegas voice extractor (Chasm game profile).

For each character listed in profile.json, resolve the NPC's exact voice from the
game data and write voices/<name>/reference.wav:

  * Named NPC  -> their own lines only, isolated from the (possibly shared) voice
                  type folder via dialogue INFO form-ids, stitched together.
  * Generic    -> a representative clip from the voice-type folder (profile
                  'voicetype' override; used for un-named NPCs like settlers).

Resolution uses the plugin: NPC_ -> VTCK -> VTYP (voice type), and DIAL/INFO
conditions (which reference the NPC's form-id) -> the INFO form-ids in filenames.

Prints "PROGRESS <name> <status>" lines for the orchestrator.
"""
import argparse, io, json, os, struct, sys, winreg, zlib

try:
    import numpy as np
    import soundfile as sf
    HAVE_SF = True
except Exception:
    HAVE_SF = False


def find_game(app_name):
    roots = []
    for hive, key in [(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam"),
                      (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Valve\Steam"),
                      (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Valve\Steam")]:
        try:
            with winreg.OpenKey(hive, key) as k:
                roots.append(winreg.QueryValueEx(k, "InstallPath")[0])
        except OSError:
            pass
    libs = list(roots)
    for r in roots:
        vdf = os.path.join(r, "steamapps", "libraryfolders.vdf")
        if os.path.exists(vdf):
            for line in open(vdf, encoding="utf-8", errors="ignore"):
                if line.strip().startswith('"path"'):
                    libs.append(line.split('"')[3].replace("\\\\", "\\"))
    for lib in dict.fromkeys(libs):
        cand = os.path.join(lib, "steamapps", "common", app_name)
        if os.path.isdir(cand):
            return cand
    return None


def subrecords(buf):
    pos, real = 0, None
    while pos + 6 <= len(buf):
        st = buf[pos:pos + 4]; sz = struct.unpack_from("<H", buf, pos + 4)[0]; pos += 6
        if st == b"XXXX":
            real = struct.unpack_from("<I", buf, pos)[0]; pos += sz; continue
        if real is not None:
            sz = real; real = None
        yield st, buf[pos:pos + sz]; pos += sz


def parse_esm(esm_path):
    data = open(esm_path, "rb").read()
    tes4 = struct.unpack_from("<I", data, 4)[0]
    HS = 24 if data[24 + tes4:28 + tes4] == b"GRUP" else 20

    def body(off):
        sz, fl, fid = struct.unpack_from("<IiI", data, off + 4)
        raw = data[off + HS:off + HS + sz]
        if fl & 0x40000:
            try: raw = zlib.decompress(raw[4:])
            except Exception: raw = b""
        return fid, raw, off + HS + sz

    vtyp = {}                # formid -> editor id
    npc_vtck = {}            # npc formid -> vtck formid
    by_edid, by_name = {}, {}
    infos = []
    o = HS + tes4
    while o < len(data):
        gs = struct.unpack_from("<I", data, o + 4)[0]
        lab = data[o + 8:o + 12]; s, e = o + HS, o + gs
        if lab == b"VTYP":
            p = s
            while p < e:
                fid, raw, p = body(p)
                for st, v in subrecords(raw):
                    if st == b"EDID":
                        vtyp[fid] = v.rstrip(b"\x00").decode("latin1")
        elif lab == b"NPC_":
            p = s
            while p < e:
                fid, raw, p = body(p)
                ed = nm = ""; vt = None
                for st, v in subrecords(raw):
                    if st == b"EDID": ed = v.rstrip(b"\x00").decode("latin1")
                    elif st == b"FULL": nm = v.rstrip(b"\x00").decode("latin1")
                    elif st == b"VTCK": vt = struct.unpack_from("<I", v, 0)[0]
                if ed: by_edid[ed.lower()] = fid
                if nm: by_name.setdefault(nm.lower(), fid)
                if vt is not None: npc_vtck[fid] = vt
        elif lab == b"DIAL":
            p = s
            while p < e:
                if data[p:p + 4] == b"GRUP":
                    cs = struct.unpack_from("<I", data, p + 4)[0]; q = p + HS
                    while q < p + cs:
                        fid, raw, q = body(q)
                        infos.append((fid, raw))
                    p += cs
                else:
                    p += HS + struct.unpack_from("<I", data, p + 4)[0]
        o += gs
    return {"vtyp": vtyp, "npc_vtck": npc_vtck, "by_edid": by_edid,
            "by_name": by_name, "infos": infos}


def parse_bsa(path):
    bd = open(path, "rb").read()
    (_m, _v, fo, fl, fc, filec, *_r) = struct.unpack_from("<4sIIIIIIII", bd, 0)
    o = fo; counts = []
    for _ in range(fc):
        _h, c, _x = struct.unpack_from("<QII", bd, o); o += 16; counts.append(c)
    p = o; dirs = []; recs = []
    for c in counts:
        nl = bd[p]; p += 1; dirs.append(bd[p:p + nl - 1].decode("latin1")); p += nl
        fr = []
        for _ in range(c):
            _h, sz, off = struct.unpack_from("<QII", bd, p); p += 16
            fr.append((off, sz & 0x3FFFFFFF))
        recs.append(fr)
    names = []
    for _ in range(filec):
        en = bd.index(b"\x00", p); names.append(bd[p:en].decode("latin1")); p = en + 1
    folder = {}; i = 0
    for d, fr in zip(dirs, recs):
        folder[d.lower()] = list(zip(names[i:i + len(fr)], fr)); i += len(fr)
    return bd, folder


def npc_info_formids(esm, npc_formid):
    tb = struct.pack("<I", npc_formid)
    return {f & 0xFFFFFF for f, raw in esm["infos"] if tb in raw}


def stitch(bd, files, out_path, limit=8):
    """Stitch the largest `limit` clips (longest lines) into one wav."""
    files = sorted(files, key=lambda x: x[1], reverse=True)[:limit]
    if not HAVE_SF:
        # No audio lib: just dump the single largest ogg.
        if files:
            off, sz = files[0][0], files[0][1]
            open(out_path.replace(".wav", ".ogg"), "wb").write(bd[off:off + sz])
        return len(files), 0.0
    chunks, sr = [], None
    for (off, sz) in [(f[0], f[1]) for f in files]:
        audio, s = sf.read(io.BytesIO(bd[off:off + sz]))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sr = s; chunks.append(audio)
    if not chunks:
        return 0, 0.0
    out = np.concatenate(chunks)
    sf.write(out_path, out, sr)
    return len(chunks), len(out) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--game-dir", default="")
    args = ap.parse_args()

    profile = json.load(open(args.profile, encoding="utf-8"))
    voice = profile["voice"]
    game = args.game_dir or find_game(voice.get("steam_app", "Fallout New Vegas"))
    if not game:
        print("ERROR game not found", flush=True); sys.exit(2)
    root = voice["voice_root"]

    print("INFO parsing plugin + voice archive ...", flush=True)
    esm = parse_esm(os.path.join(game, "Data", voice["plugin"]))
    bd, folder = parse_bsa(os.path.join(game, voice["bsa_relative"]))

    for ch in profile.get("characters", []):
        name = ch["name"]
        odir = os.path.join(args.out, name); os.makedirs(odir, exist_ok=True)
        out = os.path.join(odir, "reference.wav")

        # Resolve the NPC + voice type.
        npc_fid = None
        if ch.get("edid"):
            npc_fid = esm["by_edid"].get(ch["edid"].lower())
        if npc_fid is None:
            npc_fid = esm["by_name"].get(name.lower())
        voicetype = ch.get("voicetype") or (
            esm["vtyp"].get(esm["npc_vtck"].get(npc_fid)) if npc_fid else None)
        if not voicetype:
            print(f"PROGRESS {name} no-voice-type", flush=True); continue

        files = folder.get((root + "\\" + voicetype).lower(), [])
        oggs = [(off, sz, n) for (n, (off, sz)) in files if n.endswith(".ogg")]

        picked = []
        if npc_fid is not None:
            refs = npc_info_formids(esm, npc_fid)
            for off, sz, n in oggs:
                parts = n.rsplit("_", 2)
                if len(parts) == 3:
                    try: ih = int(parts[1], 16) & 0xFFFFFF
                    except ValueError: continue
                    if ih in refs:
                        picked.append((off, sz))
        if not picked:
            # Generic / unresolved: representative folder clips (skip tiny ones).
            picked = [(off, sz) for off, sz, n in oggs if sz > 20000]

        n_used, secs = stitch(bd, picked, out)
        tag = "own-lines" if npc_fid is not None and picked else "generic"
        print(f"PROGRESS {name} {voicetype} {tag} {n_used}clips {secs:.1f}s", flush=True)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()

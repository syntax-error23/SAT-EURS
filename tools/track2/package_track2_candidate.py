import os, sys, zipfile, hashlib

def package_track2_candidate():
    os.makedirs("deploy/track2", exist_ok=True)
    src_onnx = "deploy/EXP-COMBINED-BEST.onnx"
    out_zip = "deploy/track2/submission_track2_exp_combined_best.zip"
    
    if not os.path.exists(src_onnx):
        raise FileNotFoundError(f"Source ONNX not found: {src_onnx}")
        
    print(f"Creating Track 2 submission archive: {out_zip} from {src_onnx}...")
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        # Write model.onnx at root level of zip
        zf.write(src_onnx, arcname="model.onnx")
        
    zip_size = os.path.getsize(out_zip)
    with open(out_zip, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
        
    print(f"Archive Created Successfully:")
    print(f"  Path:     {out_zip}")
    print(f"  Size:     {zip_size:,} bytes ({zip_size / (1024*1024):.2f} MB)")
    print(f"  SHA-256:  {sha256}")
    
    # Verify archive contents
    with zipfile.ZipFile(out_zip, "r") as zf:
        infolist = zf.infolist()
        print(f"  Archive Contents ({len(infolist)} files):")
        for info in infolist:
            print(f"    - {info.filename}: {info.file_size:,} bytes (compressed: {info.compress_size:,} bytes)")
            assert info.filename == "model.onnx", f"Invalid filename in zip: {info.filename}"
            
    print("Verification PASSED!")
    return {
        "path": out_zip,
        "size": zip_size,
        "sha256": sha256
    }

if __name__ == "__main__":
    package_track2_candidate()

#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Tải MVTec AD 2 (8 category) SONG SONG, mỗi file xong tự giải nén rồi xóa zip.
# Chạy trên server thuê:
#     cd /workspace/data
#     bash /workspace/.../download_mvtecad2.sh
# An toàn: chỉ rm .tar.gz khi tar giải nén THÀNH CÔNG. wget -c để resume nếu đứt mạng.
# Log riêng từng category ở ./_dl_logs/<cat>.log
# -----------------------------------------------------------------------------
set -u

# Thư mục đích: đối số 1, hoặc mặc định /workspace/data. Tự tạo rồi cd vào.
DEST="${1:-/workspace/data}"
mkdir -p "$DEST"
cd "$DEST" || { echo "Không vào được $DEST"; exit 1; }
echo "Tải vào: $(pwd)"

URLS=(
  "https://www.mydrive.ch/shares/121501/26456e2f3ef813930866f8f9b072593a/download/466651130-1743159807/can.tar.gz"
  "https://www.mydrive.ch/shares/150999/2c2026421bcf68e37e9268885d355081/download/466651519-1743162446/fabric.tar.gz"
  "https://www.mydrive.ch/shares/121503/951a46ce30a3af3787ce9671cfa8613a/download/466651800-1743164023/fruit_jelly.tar.gz"
  "https://www.mydrive.ch/shares/121504/0014676292c3c44931712a54fb3bdbe8/download/466653907-1743164943/rice.tar.gz"
  "https://www.mydrive.ch/shares/121505/2d8fcdc8e988456bdd18696746eda0a0/download/466654829-1743166795/sheet_metal.tar.gz"
  "https://www.mydrive.ch/shares/121506/739dc6459c939fe464c0d26acc6c2d55/download/466654885-1743167505/vial.tar.gz"
  "https://www.mydrive.ch/shares/121507/66fe6e114b498e03be8d48c711794be7/download/466655287-1743168151/wallplugs.tar.gz"
  "https://www.mydrive.ch/shares/121508/9fcf67e49f0dc61a9608f57ba0482356/download/466656233-1743168988/walnuts.tar.gz"
)

LOGDIR="./_dl_logs"
mkdir -p "$LOGDIR"

fetch_one() {
  url="$1"
  fname="$(basename "$url")"      # can.tar.gz ...
  cat="${fname%.tar.gz}"
  log="$LOGDIR/$cat.log"
  {
    echo "[$(date +%T)] START $cat"
    # -c: resume; --tries: retry; --timeout tránh treo vô hạn
    wget -c --tries=5 --timeout=60 --retry-connrefused -O "$fname" "$url"
    if [ $? -ne 0 ]; then
      echo "[$(date +%T)] ✗ DOWNLOAD FAIL $cat -> giữ lại $fname để chạy lại (-c resume)"
      exit 1
    fi
    # kiểm tra gzip toàn vẹn trước khi giải nén
    if ! gzip -t "$fname" 2>/dev/null; then
      echo "[$(date +%T)] ✗ GZIP CORRUPT $cat -> tải chưa xong, giữ lại $fname"
      exit 1
    fi
    echo "[$(date +%T)] extracting $cat ..."
    if tar -xf "$fname"; then
      rm -f "$fname"
      echo "[$(date +%T)] ✓ DONE $cat (đã giải nén + xóa zip)"
    else
      echo "[$(date +%T)] ✗ TAR FAIL $cat -> giữ lại $fname"
      exit 1
    fi
  } >>"$log" 2>&1
}

echo "Bắt đầu tải song song ${#URLS[@]} category vào: $(pwd)"
echo "Xem tiến độ: tail -f $LOGDIR/*.log"

pids=()
for url in "${URLS[@]}"; do
  fetch_one "$url" &
  pids+=("$!")
done

# chờ tất cả, đếm fail
fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=$((fail+1))
done

echo "-------------------------------------------"
if [ "$fail" -eq 0 ]; then
  echo "✓ TẤT CẢ XONG. Kiểm tra: ls -d */"
else
  echo "✗ $fail category LỖI. Xem log trong $LOGDIR/ rồi CHẠY LẠI script (wget -c sẽ resume)."
fi

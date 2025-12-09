
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 

from Hardware_Model.Buffer import buffer
from Hardware_Model.PE import PE
from Hardware_Model.Tile import tile

buf = buffer(SimConfig_path="/home/zxf1/master_code/SimConfig.ini",
             buf_level=2,
             default_buf_size=16)      # 16 KB PE输入缓冲
buf.calculate_buf_write_latency(1024)   # 写 1 KB
buf.calculate_buf_read_latency(512)      # 读 512 B
print("写 1 KB 延迟 :", buf.buf_wlatency, "ns")
print("读 512 B 延迟:", buf.buf_rlatency, "ns")
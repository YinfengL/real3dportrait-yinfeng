使用指南：

服务器在Autodl，
名为Real3DPortrait新克隆，
受限于平台限制，
端口化仅可以使用运行设备终端、命令行通过SSH链接：
ssh -CNg -L 8000:127.0.0.1:8000 root@connect.cqa1.seetacloud.com -p 39909
JZQg3VET7DY/
第二行密码输入后，无显示即为正常


后端
#虚拟环境
source /root/miniconda3/bin/activate real3dportrait
#清理缓存
find . -type d -name __pycache__ -exec rm -rf {} +
find . -name "*.pyc" -delete
#删后台
ps -u $USER
pkill -f inference/app_real3dportrait.py


#启动 API 服务：
source /root/miniconda3/bin/activate real3dportrait
cd /root/Real3DPortrait
python main_api.py


#请求服务，由于使用预加载缓存，首次使用新资料速度较慢
curl -X POST "http://localhost:8000/generate-batch" \
     -H "Content-Type: application/json" \
     -d '{
           "src_image_path": "/root/Real3DPortrait/test/test.png",
           "text_input": "你好。",
           "drv_pose_path": "data/raw/examples/May_5s.mp4",
           "out_mode": "final"
         }'


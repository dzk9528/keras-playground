## Use pretrained yolo model detect and track

#### Dependencies
PIL, Tensorflow(as keras backend), keras, numpy, h5py, [yad2k](https://github.com/allanzelener/YAD2K)

#### My system configration
Win10 + GTX960m + Python 3.5

#### Notice
1. The detect_and_track function is in the module of track_sys.py along with other useful functions. yolo_utils.py is the same in the courses of deep learning specialization, which is very useful.

2. The whole Idea is using pre-trained YOLO v2 on coco datasets as the detection model, and use simple logic to realize tracking based on detection. I used yolo_model_config to load the model in keras. The pretrained YOLO model, class names are generated and transformed using the [yad2k](https://github.com/allanzelener/YAD2K) .<br />

3. If you want to test your image sequnce and see the track_sys's result: <br />

    python model_test or python model_test --data_path /path_in/--output_path /path_out/
4. If you want to open web camera to see the track result:(press 'g' to start detection)

    python yolo_with_web_cam.py
    



{
  "id": "d9564b79-1f1a-4ac4-891d-2767a0d6bc95",
  "meta": {
    "name": "YOLOv3",
    "application_area": "Object Detection",
    "task": "Object Detection",
    "task_extended": "Object Detection",
    "data_type": "Image/Photo",
    "data_source": "http://cocodataset.org/#download"
  },
  "publication": {
    "title": "YOLOv3: An Incremental Improvement",
    "source": "arXiv",
    "year": 2018,
    "authors": "Joseph Redmon, Ali Farhadi",
    "email": "pjreddie@uw.edu",
    "abstract": "We present some updates to YOLO! We made a bunch of little design changes to make it better. We also trained this new network that's pretty swell. It's a little bigger than last time but more accurate. It's still fast though, don't worry. At 320x320 YOLOv3 runs in 22 ms at 28.2 mAP, as accurate as SSD but three times faster. When we look at the old .5 IOU mAP detection metric YOLOv3 is quite good. It achieves 57.9 mAP@50 in 51 ms on a Titan X, compared to 57.5 mAP@50 in 198 ms by RetinaNet, similar performance but 3.8x faster. As always, all the code is online at this https URL",
    "url": "https://arxiv.org/abs/1804.02767",
    "google_scholar": "https://scholar.google.com/scholar?cites=12589619088479868341&as_sdt=40000005&sciodt=0,22&hl=en",
    "bibtex": "@article{DBLP:journals/corr/abs-1804-02767, author    = {Joseph Redmon and Ali Farhadi}, title     = {YOLOv3: An Incremental Improvement},journal   = {CoRR}, volume    = {abs/1804.02767}, year      = {2018}, url       = {http://arxiv.org/abs/1804.02767}, archivePrefix = {arXiv}, eprint    = {1804.02767}, timestamp = {Mon, 13 Aug 2018 16:48:24 +0200}, biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1804-02767}, bibsource = {dblp computer science bibliography, https://dblp.org}}"
  },
  "model": {
    "description": "You only look once (YOLO) is a state-of-the-art, real-time object detection system. ",
    "provenance": "https://github.com/experiencor/keras-yolo3",
    "architecture": "Convolutional Neural Network (CNN)",
    "learning_type": "Supervised Learning",
    "format": ".h5",
    "io": {
      "input": {
        "format": ["image/png", "image/jpg", "image/jpeg"],
        "dim_limits": [
          {
            "min": 1,
            "max": 5
          },
          {
            "min": 200
          },
          {
            "min": 200
          }
        ],
        "description": "Min input size for network input given"
      },
      "output": [
        {
          "name": "BB Locations and Class Probabilities",
          "type": "custom",
          "description": "A (13,13,255) numpy array where 255 = 3[Bounding Boxes per Cell]*(4[x,y,w,h of BB]+1[BB confidence score]+80[class probabilities] for each of the 13x13 grid cells. More desc in the sandbox.ipynb and here: https://towardsdatascience.com/review-yolov3-you-only-look-once-object-detection-eab75d7a1ba6"
        }
      ]
    }
  },
  "modelhub": {
    "top": 5,
    "sort": true
  }
}

from torchvision import transforms
from utils import *
import cv2
import time
import torch
import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'checkpoint_dw_2_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
model = checkpoint['model']
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_pil = transforms.ToPILImage()


def detect(original_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a cv2 Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    image = normalize(to_tensor(resize(to_pil(original_image))))

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    
    #Get height and width of original image or input image
    original_height, original_width = original_image.shape[:2]

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_width, original_height, original_width, original_height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return cv2.cvtColor(original_image,cv2.COLOR_RGB2BGR)

    # Annotate
    annotated_image = cv2.cvtColor(original_image,cv2.COLOR_RGB2BGR)
    
    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        box_location = [int(b) for b in box_location]
        annotated_image = cv2.rectangle(annotated_image, (box_location[0],box_location[1]),
                        (box_location[2],box_location[3]),label_BGR_map[det_labels[i]],2)
        
        if min(original_height,original_width)>1000:
            fontscale=0.9
        else:
            fontscale=0.35
        (text_width, text_height), baseline = cv2.getTextSize(det_labels[i].upper(),
                                                              cv2.FONT_HERSHEY_TRIPLEX,
                                                              fontscale, 1)
        annotated_image = cv2.rectangle(annotated_image, (box_location[0],box_location[1]),
                        (box_location[0]+text_width+baseline, box_location[1]-text_height-2*baseline),label_BGR_map[det_labels[i]],-1)
        annotated_image = cv2.putText(annotated_image, det_labels[i].upper(),
                                        (box_location[0],box_location[1]-text_height+baseline), cv2.FONT_HERSHEY_TRIPLEX, fontscale, 
                                        (255,255,255),1,cv2.LINE_AA)
    #del draw

    return annotated_image

def image_analysis(img_files):
    for f in (img_files):
        path = os.path.join(dir_path,f)
        original_image = cv2.imread(path,1)
        original_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
        cv2.imshow('demo_image',cv2.resize(detect(original_image, min_score=0.15, max_overlap=0.35, top_k=200),None, 
                                            fx=1.5, fy=1.5,interpolation=cv2.INTER_CUBIC))
        k = cv2.waitKey(3500)
        if k == 27:
            break        # wait for ESC key to exit
    cv2.destroyAllWindows()

    
def video_analysis(video_path):
    video_capture = cv2.VideoCapture(video_path)
    original_width = int(video_capture.get(3))
    original_height = int(video_capture.get(4))
    out = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc('m ','p','4','v'), 15, (original_width,original_height))
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret == True:
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            canvas = detect(frame, min_score=0.20, max_overlap=0.35, top_k=200)
            out.write(canvas)
            cv2.imshow('Video',canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'): # To stop the loop.
                break
        else:
            break
    video_capture.release() # We turn the webcam/video off.
    cv2.destroyAllWindows()


if __name__ == '__main__':
    dir_path = 'VOC2007/JPEGImages'
    img_files = ['000378.jpg', '000793.jpg', '001394.jpg', '001805.jpg', '000001.jpg', '000022.jpg', '000029.jpg', 
                    '000045.jpg', '000062.jpg', '000069.jpg', '000075.jpg', '000082.jpg', '000085.jpg', '000092.jpg', 
                    '000098.jpg', '000100.jpg', '000116.jpg', '000124.jpg', '000127.jpg', '000128.jpg', '000139.jpg', 
                    '000144.jpg', '000145.jpg']
    video_path = 'img/sample_video4.mp4'

    image_analysis(img_files) #Press 'ESC' key in your keyboard to end image slideshow
    video_analysis(video_path) #Press 'Q' key in your keyboard to end video
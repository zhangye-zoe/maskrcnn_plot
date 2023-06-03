import matplotlib.pyplot as plt
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
img = rgb_img.tensors[0]
img = torch.permute(img, (1,2,0)).cpu()
image = np.array(img).astype(np.uint8)

anchor = torch.cat([a.bbox for a in detection_rgb], dim=0)
boxcoder = BoxCoder(weights=(1.0,1.0,1.0,1.0))
boxes = boxcoder.decode(box_regression_rgb[:,4:], anchor)
print(prop)
for box in boxes:
    box = box.to(torch.int64)
    top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
    image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), 255, 2)
cv2.imwrite("zz_new_detection_rgb.png",image)

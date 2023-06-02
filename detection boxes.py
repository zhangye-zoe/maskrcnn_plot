box = box.to(torch.int64)
top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
image = cv2.rectangle(
    image, tuple(top_left), tuple(bottom_right), tuple(color), 1
    )

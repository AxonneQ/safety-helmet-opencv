

def combineBoundingBox(box1, box2):
    x = box1[0] if box1[0] < box2[0] else box2[0]
    y = box1[1] if box1[1] < box2[1] else box2[1]
    w = box1[2] if box1[2] > box2[2] else box2[2]
    h = box1[3] if box1[3] > box2[3] else box2[3]

    return (x, y, w, h)

def touchingRect(box1, box2):
    if box1[0] < box2[0] + box2[2] and \
    box1[0] + box1[2] > box2[0] and \
    box1[1] < box2[1] + box2[3] and \
    box1[1] + box1[3] > box2[1]:
        return True
    else:
        return False


box1 = (60, 20, 60, 60)
box2 = (10, 10, 500, 50)

print(touchingRect(box1, box2))

x, y, w, h = combineBoundingBox(box1,box2)

print(x, y, w, h)
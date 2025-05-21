import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pymunk

image = 255-cv.imread('shapes.jpg', cv.IMREAD_GRAYSCALE)
_, thresh = cv.threshold(image, 127, 255, 0)
comps = cv.connectedComponents(thresh)

def get_shapes(comps, height):
    n_shapes = comps[0]-1
    shapes = []
    for s_i in range(1, n_shapes+1):
        shape = {}
        points = np.hstack([column.reshape(-1, 1) for column in np.where((comps[1]==s_i) > 0)])
        transformed = np.zeros_like(points, float)
        transformed[:, 0] = points[:, 1]
        transformed[:, 1] = height-points[:, 0]
        center = pymunk.Poly(None, transformed.tolist()).center_of_gravity
        center = np.array([center[0], center[1]]).reshape(1, 2)
        transformed -= center
        shape['center'] = center
        shape['points'] = transformed
        shapes.append(shape)
    return shapes

shapes = get_shapes(comps, image.shape[0])

def create_bodies(shapes):
    bodies = []
    for shape in shapes:
        body = {}
        body['body'] = pymunk.Body()
        body['body'].position = shape['center'][0,0], shape['center'][0,1]
        body['poly'] = pymunk.Poly(body['body'], shape['points'].tolist())
        body['poly'].mass = body['poly'].area/50
        bodies.append(body)
    return bodies
        
bodies = create_bodies(shapes)

space = pymunk.Space()
space.gravity = 0,-981


walls = pymunk.Body(body_type = pymunk.Body.STATIC) # 1
walls.position = (0, 0)
w1 = pymunk.Segment(walls, (0, 0), (image.shape[1], 0), 2)
w2 = pymunk.Segment(walls, (0, 0), (0, image.shape[0]), 2)
w3 = pymunk.Segment(walls, (image.shape[1], 0), (image.shape[1], image.shape[0]), 2) 
w1.friction = 1
w2.friction = 1
w3.friction = 1
space.add(walls, w1, w2, w3)

for i in range(len(bodies)):
    space.add(bodies[i]['body'], bodies[i]['poly'])

def generate_frame(step):
    plt.figure(figsize=(7,9))
    for i in range(len(bodies)):
        vertices = bodies[i]['poly'].get_vertices()
        theta = bodies[i]['body'].angle
        vertices.append(vertices[0])
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        vertices = (R @ np.array(vertices).T).T
        points = vertices + np.array(bodies[i]['body'].position)
        plt.plot(points[:, 0], points[:, 1], c='black')

    plt.xlim((0, image.shape[1]))
    plt.ylim((0, image.shape[0]))

    plt.axis('off')
    plt.savefig('frames/{:04d}.png'.format(step), bbox_inches='tight', transparent=False, pad_inches=0)
    plt.close()
     
steps = 240
for step in range(steps):
    print('Processing step {} ...'.format(step))
    space.step(0.015)
    generate_frame(step)

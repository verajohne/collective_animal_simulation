from Tkinter import *
from node import Graph
from node import Node
from node import *
import numpy as np
import random 
import time

GUI_WIDTH = 800
GUI_HEIGHT = 800
CANVAS_HEIGHT = 500
CANVAS_WIDTH = 500
SLEEP_TIME = 0.25
ZOOM = 5
RADIUS = 0.5


def cartesian_to_canvas(cartesian_position_vector):
	canvasX = ZOOM*cartesian_position_vector[0] + CANVAS_WIDTH/2
	canvasY = CANVAS_HEIGHT/2 - ZOOM*cartesian_position_vector[1]
	coords = [canvasX, canvasY]
	return coords

def start_button():

	''' Create NUMBER_OF_NODES nodes '''
	#for i in range(NUMBER_OF_NODES):
	timer = 0
	nodes = []
	result = []

	for i in range(NUMBER_OF_NODES):
		x = random.uniform(-10,10)
		y = random.uniform(-10,10)
		orientation = random.uniform(0,np.pi)	#should perhaps all start within range of pi radians?
		position = [x,y]
		new_node = Node(position = position, orientation = orientation)
		nodes.append(new_node)

	g = Graph(nodes=nodes)
	g_state = copy.deepcopy(g)
	head = graph_state(graph = g_state, time = timer)
	tail = head
	'''
	should I do a few updates every time step
	'''
	'''
	while(timer < RUN_TIME):
		for node in g.nodes:	
			g.update_graph(node = node)
			g_state = copy.deepcopy(g)
			temp = graph_state(graph = g_state, time = timer)
			tail.next = temp
			tail = temp
			timer+=1
	'''
	while(timer < RUN_TIME):
		count = 1
		nodec = 1
		print "" 
		for node in g.nodes:
			print nodec
			nodec +=1
			g.update_graph(node = node)
			if count % 4:
				g_state = copy.deepcopy(g)
				temp = graph_state(graph = g_state, time = timer)
			tail.next = temp
			tail = temp
			timer+=1
			count +=1

	#canvas.delete("all")
	#canvas.pack()
	while (head.next != None):
		graph = head.graph
		for node in graph.nodes:
			canvas_coord = cartesian_to_canvas(node.position_vector)
			print node.position_vector, canvas_coord
			#canvas.create_image(canvas_coord[0],canvas_coord[1], image = label.image)
			canvas.create_oval(canvas_coord[0] - RADIUS,canvas_coord[1] - RADIUS,canvas_coord[0] + RADIUS,canvas_coord[1] + RADIUS)
			canvas.pack()
		print ""
		head = head.next
		GUI.update()
		time.sleep(SLEEP_TIME)
		canvas.delete(ALL)
		#canvas.move("all", 100,100)

	return



'''Initialisation'''
GUI = Tk()
GUI.geometry("600x600")
GUI.title("SHEEP SIMULATION")
startButton = Button(GUI,text = "START", command = start_button).pack()
canvas = Canvas(GUI, heigh = CANVAS_HEIGHT,width = CANVAS_WIDTH,bg="green")
canvas.pack()
GUI.mainloop()




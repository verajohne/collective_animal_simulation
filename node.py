from __future__ import division

####################################################

import numpy as np
import random
from math import sqrt
import copy


####################################################
#assuming uniform speed

SEPERATION_TRESHOLD = 2

SPEED = 10
NUMBER_OF_NODES = 15
#ANGLE_TRESHOLD = np.pi/12
RUN_TIME = 1000
STIME = 0.1
KNN_K = 4
VISUAL_FIELD = (270)*(np.pi/180)
REPULSE_MAGNITUDE = 1
ATTRACTION_MAGNITUDE = 1

FLOCK_RADIUS = 5


####################################################
def distance_between_points(node_0, node_1):
	dist = sqrt((node_0[0] - node_1[0])**2 + (node_0[1] - node_1[1])**2)
	return dist

def distance_between_nodes(node_0, node_1):
	dist = sqrt( ((node_0.position_vector[0]) - (node_1.position_vector[0]))**2 + ((node_0.position_vector[1]) - (node_1.position_vector[1]))**2 )
	return dist

def is_within_flock_radius(node, center_of_mass, radius = FLOCK_RADIUS):
	dist = distance_between_points(node.position_vector, center_of_mass)
	#b = dist < radius
	return dist < radius

def angele_between_nodes(o1, o2):
	#o = orientation
	v1 = np.array([np.cos(o1), np.sin(o1)])
	v2 = np.array([np.cos(o2), np.sin(o2)])
	angle = np.arccos((np.dot(v1,v2)))
	return angle

def visionary_field(orientation):
	return [(orientation - VISIONARY_FIELD/2), ((orientation + VISIONARY_FIELD/2))]

def position_vector_to_angle(unit_vector):
	x = unit_vector[0]
	y = unit_vector[1]
	phi = 0
	if x == 0:
		if y  > 0:
			phi = np.pi/2
		else:
			phi = np.pi*3/2
	else:
		temp = y/x
		temp = np.arctan(temp)
		if x < 0:
			phi = np.pi + temp
		else:
			phi = temp
	return phi


####################################################
'''TODO::
FIX ATTRACTION FUNCTION

'''

class Node:
	def __init__(self, position, orientation, speed = SPEED, graph = None):
		'''orientation in radians'''
		self.position = position
		self.speed = speed
		self.orientation = orientation
		self.graph = graph
		self.position_vector = np.array([position[0],position[1]])
		
		if orientation > 2*np.pi:
			raise Exception('Please provide orientation in radians!')

	def move_node(self, knn, knn_repulse, nodes, isInFlock, com):
		print "mode_node..."

		new_direction = np.zeros(2)
		factor = 1

		if (len(knn_repulse) == 0) & (len(knn) == 0):
			#no near neighbors, orient to see flock
			self.orientation = self.orient(nodes)
			return
		if len(knn_repulse) > 0:
			new_direction = self.repulse(knn_repulse)
			factor = 2
		else:
			if isInFlock == True:
				com = None
			new_direction = self.attract(knn, com)
		self.orientation = position_vector_to_angle(new_direction)
		self.position_vector += (factor*(self.speed)*STIME*new_direction)
		return


	def repulse(self, knn_repulse):
		repulse = np.zeros(2)
		for node in knn_repulse:
			#v = np.absolute((node.position_vector - self.position_vector))
			v = (node.position_vector - self.position_vector)
			norm = np.linalg.norm(v)
			v = v/norm
			intensity = 1/distance_between_nodes(self, node)
			v = intensity*v
			repulse += v
		temp = np.array([-1,-1])
		#repulse *= temp
		repulse /= np.linalg.norm(repulse)
		return repulse

	def attract(self, knn_attract, com):
		''' returns unit vector of '''
		attract = np.zeros(2)
		for node in knn_attract:
			v = node.position_vector - self.position_vector
			norm = np.linalg.norm(v)
			v = v/norm
			intensity = 1/distance_between_nodes(self, node)
			v = intensity*v
			attract += v
		if com != None:
			v = com - self.position_vector
			dist = distance_between_points(com, self.position_vector)
			if dist != 0:
				v = v/norm
				intensity = 2*(1/dist)
				attract += v
		#attract = attract*(1/len(knn_attract))
		attract /= np.linalg.norm(attract)

		return attract

	def orient(self, knn):
		'''orients to the avg orientation of knn'''
		avg_orientation = 0
		for node in knn:
			avg_orientation += node.orientation
		avg_orientation = avg_orientation/KNN_K
		return avg_orientation
	
	def __eq__(self, other):
		return self.position_vector[0] == self.position_vector[0] & self.position_vector[1] == self.position_vector[1]

########################################################

class Graph:
	def __init__(self, nodes):
		self.nodes = nodes
		for node in self.nodes:
			node.graph = self

	def center_of_mass(self):
		cm = np.zeros(2)
		for node in self.nodes:
			cm += node.position_vector
		return cm


	def knn_visual(self, node):
		print "KNN VISUAL"
		knn_visual_nodes = []
		knn_repulse = []

		radius = 20
		ori = node.orientation
		sigma = ori - VISUAL_FIELD/2
		lba = sigma
		uba = sigma + VISUAL_FIELD 
		print lba, uba
 		
		for node_1 in self.nodes:
			dist = distance_between_nodes(node,node_1)
			if dist == 0:
				break
			v = node_1.position_vector - node.position_vector
			phi = position_vector_to_angle(v)

			if (phi > lba) & (phi < uba ):
				knn_visual_nodes.append(node_1)
				if dist < SEPERATION_TRESHOLD:
					knn_repulse.append(node_1)


		return [knn_visual_nodes, knn_repulse]
			
	def knn_per_node(self, node_0):
		'''	Lousy KNN algorithm implemented
		TODO: implement fast algorithm that consider VISIONARY_FIELD
		'''
		dist_node = []
		nodes = []
		nodes_repulse = []

		for node_1 in self.nodes:
			dist = distance_between_nodes(node_0,node_1)
			dn = [dist,node_1]
			if dist != 0:
				dist_node.append(dn)
				if (dist < SEPERATION_TRESHOLD):
					nodes_repulse.append(node_1)
		dist_node.sort()
		dist_node = dist_node[0:KNN_K]
		for dn in dist_node:
			nodes.append(dn[1])
		return [nodes, nodes_repulse]
			

	def update_graph(self, node):
		#for node in self.nodes:
		print "updating graph node..."
		#info = self.knn_per_node(node)
		info = self.knn_visual(node)
		knn = info[0]
		knn_repulse = info[1]
		com = self.center_of_mass()
		isInFlock = is_within_flock_radius(node = node, center_of_mass = com)
		node.move_node(knn=knn, knn_repulse = knn_repulse, nodes = self.nodes, isInFlock = isInFlock, com = com)


##########

class graph_state:
	def __init__(self, graph, time, next = None):
		self.graph = graph
		self.time = time
		self.next = next

		
			
#########################################################

def main():
	#create nodes
	timer = 0
	nodes = []
	result = []

	for i in range(NUMBER_OF_NODES):
		x = random.randrange(-10,10)
		y = random.randrange(-10,10)
		orientation = random.uniform(0,2*np.pi)	#should perhaps all start within range of pi radians?
		position = [x,y]
		new_node = Node(position = position, orientation = orientation)
		nodes.append(new_node)

	g = Graph(nodes=nodes)
	
	for node in g.nodes:
		print node.position_vector
		#print node.orientation
		print ""
	

	'''initialize linked list'''
	head = graph_state(graph = g, time = timer)
	tail = head

	while(timer < RUN_TIME):
		for node in g.nodes:	
			g.update_graph(node = node)
			g_state = copy.deepcopy(g)
			temp = graph_state(graph = g_state, time = timer)
			tail.next = temp
			tail = temp
			timer+=1

	
		
if __name__ == '__main__':
	main()







# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:40:19 2022

@author: shrut
"""
import numpy as np
import kdtree

class ObjectData(object):

    def __init__(self, x, y, id):
        self.coords = (x, y)
        self.id = id
        self.pos = None
        self.neg = None
        self.id = id
        self.angle = None

    def __getitem__(self, i):
        return self.coords[i]

    def __len__(self):
        return len(self.coords)

    def __repr__(self):
        return 'Item(x:{}, y:{}, pos:{}, neg:{}, id:{}, angle:{})'.format(self.coords[0], self.coords[1], self.pos, self.neg, self.id, self.angle)


class KD_Tree():
    def __init__(self):
        self.tree = kdtree.create(dimensions=2)
        self.nextID = 0
    
    def show(self):
        kdtree.visualize(self.tree)

    def asList(self):
        return list(self.tree.inorder())

    def addNode(self, x, y):
        # make a obj with new data
        obj_data = ObjectData(x, y, self.nextID)

        # add to tree
        self.tree.add(obj_data)

        # increment nextID counter
        self.nextID +=1

    def updateNode(self, x, y, neighbor):
        # save the neighbor's id
        id = neighbor[0].data.id

        # remove old match
        self.tree = self.tree.remove(neighbor[0].data)

        # make a obj with new coordinates and old ID
        obj_data = ObjectData(x, y, id)
        # insert obj to tree
        self.tree.add(obj_data)

    def NN(self, x, y):
        return self.tree.search_nn((x,y))
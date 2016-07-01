# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 13:13:31 2016

@author: wb
"""
import numpy as np

""" graph linking net objects
"""
class netgraph(object):
    def __init__(self):
        self.CL = dict()

        self._mode = 'tr' # tr=in train; ut=in utilization;

    def addnet(self,netname):
        # Check if already exist
        if netname in self.CL:
            print('conflict with existing node')
            return
        else:
            self.CL[netname]=[]


    def addlink(self,linfrom,linto):
        # 0 means from/to whole object
        if isinstance(linfrom,str):
            linfrom=(linfrom,0)
        if isinstance(linto,str):
            linto=(linto,0)

        self.CL[linfrom[0]][linfrom[1]].append([linto[0],linto[1]])
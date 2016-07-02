# -*- coding: utf-8 -*-
"""
@author: wb
"""
import numpy as np

# tr=in train; ut=in utilization;
MK_MODE_TRAIN='tr'
MK_MODE_UT='ut'

### Graph object linking network objects
class netgraph(object):
    def __init__(self):
        self.nets=[] # list of networks.
        """ CL structure:
        CL[net1]: connection from net1, CL[net1][0]: from all unit, CL[net1][i]: from unit set i (i>0).
        CL[net1][2]=(net2,1): connection from net1 set2 to net2 set1.
        """
        self.CL = dict()

        self._mode = MK_MODE_UT


    # Add net object to graph.
    def addnet(self,netname):
        # Check if already exist
        if netname in self.nets:
            print('conflict with existing node')
            return
        else:
            self.nets.append(netname)
            self.CL.append(netname)


    # Add inter-net link.
    def addlink(self,linfrom,linto):
        # 0 means from/to whole object
        if isinstance(linfrom,str):
            linfrom=(linfrom,0)
        if isinstance(linto,str):
            linto=(linto,0)

        if linfrom[0] not in self.nets:
            print('source net not found')
            return
        if linto[0] not in self.nets:
            print('target net not found')
            return

        self.CL[linfrom[0]][linfrom[1]].append([linto[0],linto[1]])
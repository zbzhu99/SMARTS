import sys
from direct.showbase.ShowBase import ShowBase
from pandac.PandaModules import *
from direct.directtools.DirectGeometry import LineNodePath


class Application(ShowBase):

    def __init__(self):
        ShowBase.__init__(self)

        l = LineNodePath(render2d, 'box', 4, VBase4(1, 0, 0, 1))
        p1 = (-.5, -.5, 0)
        p2 = (-.5, .5, 0)
        p3 = (.5, .5, 0)
        p4 = (.5, -.5, 0)
        l.drawLines([[p1, p2], [p2, p3], [p3, p4], [p4, p1]])
        l.create()

        ls = LineSegs()
        ls.setThickness(10)

        # X axis
        ls.setColor(1.0, 0.0, 0.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(1.0, 0.0, 0.0)

        # Y axis
        ls.setColor(0.0, 1.0, 0.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(0.0, 1.0, 0.0)

        # Z axis
        ls.setColor(0.0, 0.0, 1.0, 1.0)
        ls.moveTo(0.0, 0.0, 0.0)
        ls.drawTo(0.0, 0.0, 1.0)

        node = ls.create()
        render2d.attachNewNode(node)

        segs = LineSegs()
        segs.setThickness(2.0)
        segs.setColor(Vec4(1, 1, 0, 1))
        node = segs.create()
        render2d.attachNewNode(node)


Application().run()
# from panda3d.egg import EggData, EggVertexPool, EggVertex, EggGroup, EggLine, loadEggData, EggNurbsCurve
# from panda3d.core import Point3D, NodePath
#
#
# class createNurbsCurve():
#     def __init__(self):
#         self.data = EggData()
#         self.vtxPool = EggVertexPool('mopath')
#         self.data.addChild(self.vtxPool)
#         self.eggGroup = EggGroup('group')
#         self.data.addChild(self.eggGroup)
#         self.myverts = []
#
#     def addPoint(self, pos):
#         eggVtx = EggVertex()
#         eggVtx.setPos(Point3D(pos[0], pos[1], pos[2]))
#         self.myverts.append(eggVtx)
#         self.vtxPool.addVertex(eggVtx)
#
#     def getNodepath(self):
#         myCurve = EggNurbsCurve()
#         myCurve.setup(3, len(self.myverts) + 3)
#         myCurve.setCurveType(1)
#         for i in self.myverts:
#             myCurve.addVertex(i)
#         self.eggGroup.addChild(myCurve)
#         # self.data.writeEgg(Filename('test.egg'))
#         return NodePath(loadEggData(self.data))
#
#
# class createLine():
#     def __init__(self):
#         self.data = EggData()
#         self.vtxPool = EggVertexPool('line')
#         self.data.addChild(self.vtxPool)
#         self.eggGroup = EggGroup('group')
#         self.data.addChild(self.eggGroup)
#         self.myverts = []
#
#     def addPoint(self, pos):
#         eggVtx = EggVertex()
#         eggVtx.setPos(Point3D(pos[0], pos[1], pos[2]))
#         self.myverts.append(eggVtx)
#         self.vtxPool.addVertex(eggVtx)
#
#     def getNodepath(self):
#         for i in range(len(self.myverts)):
#             if not i % 500:
#                 if i:
#                     myline.addVertex(self.myverts[i])
#                 print(i, len(self.myverts))
#                 myline = EggLine()
#                 self.eggGroup.addChild(myline)
#             myline.addVertex(self.myverts[i])
#         # self.data.writeEgg(Filename('test.egg'))
#         return NodePath(loadEggData(self.data))
#
#
# if __name__ == '__main__':
#     from direct.directbase import DirectStart
#     from math import sin, cos
#     from direct.directutil.Mopath import Mopath
#     from direct.interval.MopathInterval import *
#     from panda3d.core import NodePath
#
#     myCurve = createNurbsCurve()
#     myLine = createLine()
#     for i in range(100):
#         myCurve.addPoint((cos(i / 3.), i, sin(i / 3.)))
#         myLine.addPoint((cos(i / 3.), i, sin(i / 3.)))
#     lineNode = myLine.getNodepath()
#     curveNode = myCurve.getNodepath()
#     lineNode.reparentTo(render)
#
#     myMopath = Mopath()
#     myMopath.loadNodePath(curveNode)
#     myMopath.fFaceForward = True
#     myCube = loader.loadModel("yup-axis")
#     myCube.reparentTo(render)
#     myInterval = MopathInterval(myMopath, myCube, duration=10, name="Name")
#     myInterval.start()
#
#     run()

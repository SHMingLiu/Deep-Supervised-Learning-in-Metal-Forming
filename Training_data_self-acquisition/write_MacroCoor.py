# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 22:46:14 2020

@author: sl7516
"""
def write_MacroCoor(pwd, geometry_name, job_name):
    fr = open('Macro_ExtractCoor.py', 'w')
    fr.write('# -*- coding: mbcs -*-\n')
    fr.write('# Do not delete the following import lines\n')
    fr.write('from abaqus import *\n')
    fr.write('from abaqusConstants import *\n')
    fr.write('import __main__\n\n')
    
    fr.write('def extract_coor():\n')
    fr.write('    import visualization\n    import xyPlot\n    import displayGroupOdbToolset as dgo\n')
    fr.write('    session.mdbData.summary()\n')
    fr.write('    o1 = session.openOdb(name=\'{}/{}.odb\')\n'.format(pwd, job_name))
    fr.write('    session.viewports[\'Viewport: 1\'].setValues(displayedObject=o1)\n')
    fr.write('    leaf = dgo.LeafFromPartInstance(partInstanceName=(\'SHEET-1\', ))\n')
    fr.write('    session.viewports[\'Viewport: 1\'].odbDisplay.displayGroup.intersect(leaf=leaf)\n')
    fr.write('    session.viewports[\'Viewport: 1\'].odbDisplay.display.setValues(plotState=(\n')
    fr.write('        CONTOURS_ON_DEF, ))\n')
    fr.write('    lastFrame=o1.steps[\'Step-Lift\'].frames[-1]\n')
    fr.write('    odb = session.odbs[\'{}/{}.odb\']\n'.format(pwd, job_name))
    fr.write('    nf = NumberFormat(numDigits=9, precision=0, format=ENGINEERING)\n')
    fr.write('    session.fieldReportOptions.setValues(numberFormat=nf)\n')
    fr.write('    session.writeFieldReport(fileName=\'{}\', append=OFF,\n'.format(geometry_name))
    fr.write('        sortItem=\'Node Label\', odb=odb, step=1, frame=lastFrame,\n')
    fr.write('        outputPosition=NODAL, variable=((\'COORD\', NODAL, ((COMPONENT, \'COOR1\'),\n')
    fr.write('        (COMPONENT, \'COOR2\'), )), ), stepFrame=SPECIFY)\n\n')
    
    fr.write('extract_coor()')
    
import xml.etree.cElementTree as ET
from xml.dom import minidom
from xml.etree import ElementTree
import os.path

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

class RSMLWriter():
    @staticmethod
    def save(key, output_dir, plants):
        root = ET.Element('rsml') 
        metadata = ET.SubElement(root, 'metadata')
        ET.SubElement(metadata,  'version').text = "1"
        ET.SubElement(metadata, 'unit').text = "pixel"
        ET.SubElement(metadata, 'resolution').text = "1"
        ET.SubElement(metadata, 'last-modified').text = "1"
        ET.SubElement(metadata, 'software').text = "ROOT_NAV.2.0"
        ET.SubElement(metadata, 'user').text = "Robi"
        ET.SubElement(metadata, 'file-key').text = key
        scene = ET.SubElement(root, 'scene')

        for plant_id, p in enumerate(plants):
            plant = ET.SubElement(scene, 'plant', id=str(plant_id+1), label="barley")
            annotations = ET.SubElement(plant, 'annotations')
            stem_angle_annotation = ET.SubElement(annotations, 'annotation', name='stem angle')
            ET.SubElement(stem_angle_annotation, 'value').text = str(p.stem_angle)
            
            seed_annotation = ET.SubElement(annotations, 'annotation', name='seed position')
            ET.SubElement(seed_annotation, 'point', x=str(p.seed[1]), y=str(p.seed[0])) 

            for primary_id, pri in enumerate(p.roots):
                priroot = ET.SubElement(plant, 'root', id=str(primary_id+1), label="primary", poaccession="1")
                geometry = ET.SubElement(priroot, 'geometry')
                polyline = ET.SubElement(geometry, 'polyline')

                spline = pri.spline
                rootnavspline = ET.SubElement(geometry, 'rootnavspline', controlpointseparation= str(spline.knot_spacing), tension=str(spline.tension))
                for c in spline.knots:
                    point = ET.SubElement(rootnavspline, 'point', x=str(c[0]), y=str(c[1]))

                poly = spline.polyline(sample_spacing = 1)
                for pt in poly:
                    point = ET.SubElement(polyline, 'point', x=str(pt[0]), y=str(pt[1]))

                functions = ET.SubElement(priroot, 'functions')
                function = ET.SubElement(functions, 'function', domain='polyline', name='diameter')
                for sample in pri.diameters:
                    s = ET.SubElement(function, 'sample', value=str(sample))
                
        tree = ET.ElementTree(root)
        rsml_text = prettify(root)

        output_path = os.path.join(output_dir, "{0}.rsml".format(key))
        with open (output_path, 'w') as f:
            f.write(rsml_text)

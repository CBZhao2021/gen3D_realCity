import glob
import cv2
import os
import trimesh
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from shapely import line_merge
from scipy.spatial import Delaunay
from shapely.affinity import rotate, scale, translate

from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from utils.tools import *
from bg_extract.bg_extract import inference_image_road, inference_image_vegetation
from bg_extract.road_centerline.road_tiff_polygonize import road_line_ext

from pleatau_inference import inference, OBJ_output, bldg_lod1_gen_realCity, bldg_citygml_realCity, save_citygml, inference
from geojson_reader import read_geojson_and_rasterize

import argparse

cfg_file = '/fast/zcb/data/PLATEAU_obj/gen3d_realCity/models/cldm_v21.yaml'
ckpt_files = ['/fast/zcb/code/ControlNet/lightning_logs/plateau_dataEnhancement_type1/checkpoints/epoch=134-step=20999.ckpt',
              '/fast/zcb/code/ControlNet/lightning_logs/plateau_dataEnhancement_type2/checkpoints/epoch=36-step=28999.ckpt',
              '/fast/zcb/code/ControlNet/lightning_logs/plateau_dataEnhancement_type3/checkpoints/epoch=17-step=25999.ckpt',
              '/fast/zcb/code/ControlNet/lightning_logs/plateau_dataEnhancement_NType5/checkpoints/epoch=95-step=18999.ckpt',
              '/fast/zcb/code/ControlNet/lightning_logs/plateau_dataEnhancement_NType6/checkpoints/epoch=13-step=13999.ckpt']


class genRelief:
    def __init__(self,
                 reilef_src_root='./data_src/src_2d/dem/crop_resize',
                 height_limit=5.,
                 **kwargs):
        self.relief_src_root = reilef_src_root
        self.height_limit = height_limit

        self.relief_src_path = glob.glob(os.path.join(self.relief_src_root, '*.jpg'))
        self.points_relief = None
        self.mesh_relief = []

    def gen_mesh_relief_lod1(self, x_min, y_min, width=200., height=200.):
        l_path = len(self.relief_src_path)
        rand_idx = random.randint(0, l_path - 1)
        rand_dem_path = self.relief_src_path[rand_idx]

        img = cv2.imread(rand_dem_path, 0)

        h_img, w_img = img.shape
        x, y = np.meshgrid(np.arange(w_img), np.arange(h_img))
        coordinates = np.stack((x_min + x * width / (w_img - 1), y_min + y * height / (h_img - 1)), axis=-1)
        points = np.concatenate([coordinates, img[..., None] / 255. * self.height_limit], axis=2)
        points_reshape = np.reshape(points, (-1, 3))

        points_2d = points_reshape[:, :2]
        tri = Delaunay(points_2d)
        faces = tri.simplices

        self.mesh_relief = trimesh.Trimesh(vertices=points_reshape, faces=faces)
        self.points_relief = points
        return self.mesh_relief

    def create_citygml_relief(self, relief, relief_lod=1, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                              srsDimension="3"):
        nsmap = {
            'core': "http://www.opengis.net/citygml/2.0",
            'dem': "http://www.opengis.net/citygml/relief/2.0",
            'gml': "http://www.opengis.net/gml"
        }

        cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

        for relief_data in relief:
            vertices, faces = relief_data.vertices, relief_data.faces

            x_max, y_max, z_max = np.max(vertices, axis=0)
            x_min, y_min, z_min = np.min(vertices, axis=0)
            boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
            Envelope = etree.SubElement(boundedBy, "{http://www.opengis.net/gml}Envelope", srsName=srs_name,
                                        srsDimension=srsDimension)
            lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
            upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
            lowerCorner.text = '{} {} {}'.format(x_min, y_min, z_min)
            upperCorner.text = '{} {} {}'.format(x_max, y_max, z_max)

            relief_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            reliefFeature = etree.SubElement(relief_member, "{http://www.opengis.net/citygml/relief/2.0}ReliefFeature")

            lod1_1 = etree.SubElement(reliefFeature, "{http://www.opengis.net/citygml/relief/2.0}lod")
            lod1_1.text = str(relief_lod)
            reliefComponent = etree.SubElement(reliefFeature,
                                               "{http://www.opengis.net/citygml/relief/2.0}reliefComponent")
            TINRelief = etree.SubElement(reliefComponent, "{http://www.opengis.net/citygml/relief/2.0}TINRelief")
            lod1_2 = etree.SubElement(TINRelief, "{http://www.opengis.net/citygml/relief/2.0}lod")
            lod1_2.text = str(relief_lod)
            tin = etree.SubElement(TINRelief, "{http://www.opengis.net/citygml/relief/2.0}tin")
            triangulatedSurface = etree.SubElement(tin, "{http://www.opengis.net/gml}TriangulatedSurface",
                                                   srsName=srs_name,
                                                   srsDimension=srsDimension)
            trianglePatches = etree.SubElement(triangulatedSurface, "{http://www.opengis.net/gml}trianglePatches")

            for face in faces:
                polygon = etree.SubElement(trianglePatches, "{http://www.opengis.net/gml}Triangle")
                exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                coords = ' '.join(
                    ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                posList.text = coords

        return cityModel

    def gen_relief_run(self, x_min, y_min, width=200., height=200., relief_lod=1, save_gml=True, gml_root=''):
        if relief_lod == 1:
            self.gen_mesh_relief_lod1(x_min, y_min, width, height)
        if save_gml and relief_lod:
            relief_gml = self.create_citygml_relief([self.mesh_relief], relief_lod=1,
                                                    srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                                                    srsDimension="3")
            save_citygml(relief_gml, os.path.join(gml_root, 'relief.gml'))
            return self.mesh_relief


class genRoad:
    def __init__(self,
                 img_path,
                 width=2.,
                 width_sub=0.1,
                 light_ratio=0.1,
                 tele_ratio=1.,
                 **kwargs):
        self.img_path = img_path
        # self.img_resolution = img_resolution
        self.width = width
        self.width_sub = width_sub
        self.light_ratio = light_ratio
        self.tele_ratio = tele_ratio

        self.road_limit = None

    def crop_road_lineStr(self):
        im_proj, im_Geotrans, im_data = read_tif(self.img_path)
        img_shape = im_data.shape
        if min(img_shape) == img_shape[2]:
            height, width, channel = im_data.shape
            im_data = im_data[:, :, :3]
        else:
            channel, height, width = im_data.shape
            im_data = np.transpose(im_data[:3, :, :], (1, 2, 0))

        x_min, y_min, resolusion_x, resolusion_y = im_Geotrans[0], im_Geotrans[3], im_Geotrans[1]*width/1024., im_Geotrans[5]*height/1024.
        road_masks = inference_image_road(im_data, 'bg_extract/ckpt/p2cnet_road.pth')

        road_link = road_line_ext(road_masks, [resolusion_x,resolusion_y], x_min, y_min)
        road_link = [line_merge(xx) for xx in road_link]
        road_link_simp = []
        
        for road_link_ in road_link:
            road_link_simp += [xx.simplify(0.5) for xx in road_link_.geoms] \
                if isinstance(road_link_, MultiLineString) else [road_link_.simplify(0.5)]
        self.roi_road = gpd.array.GeometryArray(np.array(road_link))

        return self.roi_road

    def gen_mesh_road(self, shp, buffer):
        self.mesh_road = []
        self.buffered_line = shp.buffer(buffer)
        self.road_limit = self.buffered_line

        for poly_road in self.buffered_line:
            if isinstance(poly_road, Polygon):
                tmp_mesh = polygon_to_mesh(poly_road)
                self.mesh_road.append(tmp_mesh)
            elif isinstance(poly_road, MultiPolygon):
                for poly_road_tmp in poly_road.geoms:
                    tmp_mesh = polygon_to_mesh(poly_road_tmp)
                    self.mesh_road.append(tmp_mesh)

    def gen_mesh_road_sub(self, shp, width, width_sub):
        left_sub, right_sub = [], []
        for tmp_road in shp:
            if isinstance(tmp_road, LineString):
                left_sub.append(
                    tmp_road.parallel_offset(width * (1 + width_sub + 0.5), 'left').buffer(width * width_sub))
                right_sub.append(
                    tmp_road.parallel_offset(width * (1 + width_sub + 0.5), 'right').buffer(width * width_sub))
            elif isinstance(tmp_road, MultiLineString):
                for tmp_road_ in tmp_road.geoms:
                    left_sub.append(
                        tmp_road_.parallel_offset(width * (1 + width_sub + 0.5), 'left').buffer(width * width_sub))
                    right_sub.append(
                        tmp_road_.parallel_offset(width * (1 + width_sub + 0.5), 'right').buffer(
                            width * width_sub))

        self.road_limit = shp.buffer(width * (1 + width_sub * 2. + 0.5))

        for poly_road in left_sub:
            if isinstance(poly_road, Polygon):
                tmp_mesh = polygon_to_mesh(poly_road)
                self.mesh_road.append(tmp_mesh)
            elif isinstance(poly_road, MultiPolygon):
                for poly_road_tmp in poly_road.geoms:
                    tmp_mesh = polygon_to_mesh(poly_road_tmp)
                    self.mesh_road.append(tmp_mesh)

        for poly_road in right_sub:
            if isinstance(poly_road, Polygon):
                tmp_mesh = polygon_to_mesh(poly_road)
                self.mesh_road.append(tmp_mesh)
            elif isinstance(poly_road, MultiPolygon):
                for poly_road_tmp in poly_road.geoms:
                    tmp_mesh = polygon_to_mesh(poly_road_tmp)
                    self.mesh_road.append(tmp_mesh)

    def generate_poles_along_line(self, line, interval):
        length = line.length
        return [line.interpolate(distance) for distance in range(0, int(length), interval)]

    def gen_device_lod1(self, shp):
        self.mesh_device = []

        left_sub = []
        for tmp_road in shp:
            if isinstance(tmp_road, LineString):
                left_sub.append(
                    tmp_road.parallel_offset(self.width * (1 + self.width_sub * 2 + 0.5), 'left'))
            elif isinstance(tmp_road, MultiLineString):
                for tmp_road_ in tmp_road.geoms:
                    left_sub.append(
                        tmp_road_.parallel_offset(self.width * (1 + self.width_sub * 2 + 0.5), 'left'))
        self.road_limit = shp.buffer(self.width * (1 + self.width_sub * 2. + 0.5))
        tele_pole_point = []
        for tmp_road in left_sub:
            tele_pole_point += self.generate_poles_along_line(tmp_road, 20)

        for x in range(len(tele_pole_point)):
            half_side = 0.1
            tele_pole_point_xy = [tele_pole_point[x].x, tele_pole_point[x].y]
            tele_pole_square_coords = [
                (tele_pole_point_xy[0] - half_side, tele_pole_point_xy[1] - half_side),
                (tele_pole_point_xy[0] - half_side, tele_pole_point_xy[1] + half_side),
                (tele_pole_point_xy[0] + half_side, tele_pole_point_xy[1] + half_side),
                (tele_pole_point_xy[0] + half_side, tele_pole_point_xy[1] - half_side)
            ]

            tele_pole_square = Polygon(tele_pole_square_coords)
            vertices, faces = polygon_to_mesh_3D(tele_pole_square)
            tmp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            if not tele_pole_point[x].within(self.road_limit).any():
                self.mesh_device.append(tmp_mesh)

    def gen_device_lod2(self, shp):
        self.mesh_device = []

        tele_pole_mesh = trimesh.load(os.path.join('./data_src/src_3d/lod3frn/electric_pole',
                                                   'obj_52385618_frn_6697_op_frn_0ece98a1-6070-4315-88d4-3d4546168814__493155_25.obj'))
        tele_pole_mesh = tele_pole_mesh.dump(concatenate=True) if isinstance(tele_pole_mesh,
                                                                             trimesh.Scene) else tele_pole_mesh
        tele_pole_mesh_xy = tele_pole_mesh.centroid[:2]
        tele_pole_mesh_zmin = np.min(tele_pole_mesh.vertices[:, 2])
        tele_pole_mesh_h = np.max(tele_pole_mesh.vertices[:, 2]) - tele_pole_mesh_zmin

        traf_light_mesh = trimesh.load(os.path.join(r'./data_src/src_3d/lod3frn/traffic_light',
                                                    'obj_52385618_frn_6697_op_frn_25870971-faa3-4677-b281-f192176cfbea__711063_25.obj'))
        traf_light_mesh = traf_light_mesh.dump(concatenate=True) if isinstance(traf_light_mesh,
                                                                               trimesh.Scene) else traf_light_mesh
        traf_light_mesh_xy = traf_light_mesh.centroid[:2]
        traf_light_mesh_zmin = np.min(traf_light_mesh.vertices[:, 2])

        left_sub = []
        for tmp_road in shp:
            if isinstance(tmp_road, LineString):
                left_sub.append(
                    tmp_road.parallel_offset(self.width * (1 + self.width_sub * 2 + 0.5), 'left'))
            elif isinstance(tmp_road, MultiLineString):
                for tmp_road_ in tmp_road.geoms:
                    left_sub.append(
                        tmp_road_.parallel_offset(self.width * (1 + self.width_sub * 2 + 0.5), 'left'))
        self.road_limit = shp.buffer(self.width * (1 + self.width_sub * 2. + 0.5))
        tele_pole_point = []
        for tmp_road in left_sub:
            tele_pole_point += self.generate_poles_along_line(tmp_road, 20)

        res_tele_pole = []
        for x in range(len(tele_pole_point)):
            tmp_tele_pole_mesh = tele_pole_mesh.copy()
            tele_pole_point_xy = [tele_pole_point[x].x, tele_pole_point[x].y]
            trans_tele_mesh = [tele_pole_point_xy[0] - tele_pole_mesh_xy[0],
                               tele_pole_point_xy[1] - tele_pole_mesh_xy[1],
                               -tele_pole_mesh_zmin]
            if not tele_pole_point[x].within(self.road_limit).any():
                res_tele_pole.append(tmp_tele_pole_mesh.apply_translation(trans_tele_mesh))

        res_traf_light = []
        for x in random.sample(list(range(len(res_tele_pole))),
                               int(len(res_tele_pole) * self.light_ratio / (self.light_ratio + self.tele_ratio))):
            tmp_traf_light_mesh = traf_light_mesh.copy()
            tele_pole_point_xy = res_tele_pole[x].centroid[:2]
            trans_traf_mesh = [tele_pole_point_xy[0] - traf_light_mesh_xy[0],
                               tele_pole_point_xy[1] - traf_light_mesh_xy[1],
                               tele_pole_mesh_h - traf_light_mesh_zmin]
            res_traf_light.append(tmp_traf_light_mesh.apply_translation(trans_traf_mesh))

        self.mesh_device += res_tele_pole + res_traf_light

    def create_citygml_road(self, roads, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                            srsDimension="3"):
        nsmap = {
            'core': "http://www.opengis.net/citygml/2.0",
            'tran': "http://www.opengis.net/citygml/transportation/2.0",
            'gml': "http://www.opengis.net/gml"
        }

        cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

        total_vertices = []
        for road in roads:
            total_vertices.append(road.vertices)
        total_vertices = np.vstack(total_vertices)
        x_max, y_max, z_max = np.max(total_vertices, axis=0)
        x_min, y_min, z_min = np.min(total_vertices, axis=0)
        boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
        Envelope = etree.SubElement(boundedBy, "{http://www.opengis.net/gml}Envelope", srsName=srs_name,
                                    srsDimension=srsDimension)
        lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
        upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
        lowerCorner.text = '{} {} {}'.format(x_min, y_min, z_min)
        upperCorner.text = '{} {} {}'.format(x_max, y_max, z_max)

        for road_data in roads:
            vertices, faces = road_data.vertices, road_data.faces

            traffic_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            transportation = etree.SubElement(traffic_member,
                                              "{http://www.opengis.net/citygml/transportation/2.0}Road")

            lod1MultiSurface = etree.SubElement(transportation,
                                                "{http://www.opengis.net/citygml/transportation/2.0}lod2MultiSurface")
            multiSurface = etree.SubElement(lod1MultiSurface, "{http://www.opengis.net/gml}MultiSurface")

            for face in faces:
                surfaceMember = etree.SubElement(multiSurface, "{http://www.opengis.net/gml}surfaceMember")
                polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                coords = ' '.join(
                    ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                posList.text = coords

        return cityModel

    def create_citygml_cityfurniture(self, devices, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                                     srsDimension="3"):
        nsmap = {
            'core': "http://www.opengis.net/citygml/2.0",
            'frn': "http://www.opengis.net/citygml/cityfurniture/2.0",
            'gml': "http://www.opengis.net/gml"
        }

        cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

        total_vertices = []
        for device in devices:
            total_vertices.append(device.vertices)
        total_vertices = np.vstack(total_vertices)
        x_max, y_max, z_max = np.max(total_vertices, axis=0)
        x_min, y_min, z_min = np.min(total_vertices, axis=0)
        boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
        Envelope = etree.SubElement(boundedBy, "{http://www.opengis.net/gml}Envelope", srsName=srs_name,
                                    srsDimension=srsDimension)
        lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
        upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
        lowerCorner.text = '{} {} {}'.format(x_min, y_min, z_min)
        upperCorner.text = '{} {} {}'.format(x_max, y_max, z_max)

        for devices_data in devices:
            vertices, faces = devices_data.vertices, devices_data.faces

            furniture_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            cityFurniture = etree.SubElement(furniture_member,
                                             "{http://www.opengis.net/citygml/cityfurniture/2.0}CityFurniture")

            lod1Geometry = etree.SubElement(cityFurniture,
                                            "{http://www.opengis.net/citygml/cityfurniture/2.0}lod1Geometry")
            multiSurface = etree.SubElement(lod1Geometry, "{http://www.opengis.net/gml}MultiSurface")

            for face in faces:
                surfaceMember = etree.SubElement(multiSurface, "{http://www.opengis.net/gml}surfaceMember")
                polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                coords = ' '.join(
                    ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                posList.text = coords

        return cityModel

    def add_relief(self, points_relief):
        z_points_interpolate = relief_interpolate(self.mesh_road, points_relief)
        if (z_points_interpolate is None):
            return

        for i, tmp_mesh in enumerate(self.mesh_road):
            tmp_vertices = tmp_mesh.vertices
            tmp_vertices[:, 2] += z_points_interpolate[i] + 0.01
            tmp_mesh.vertices = tmp_vertices

    def gen_road_run(self, road_lod=1, device_lod=2, points_relief=None, save_gml=True, gml_root=''):
        if road_lod == 1:
            self.gen_mesh_road(self.roi_road, self.width)
            self.gen_mesh_road_sub(self.roi_road, self.width, self.width_sub)
        if device_lod == 1:
            self.gen_device_lod1(self.roi_road)
        elif device_lod == 2:
            self.gen_device_lod2(self.roi_road)
        road_ori = self.mesh_road.copy()
        self.mesh_road += self.mesh_device
        self.add_relief(points_relief)
        if save_gml:
            road_gml = self.create_citygml_road(road_ori)
            save_citygml(road_gml, os.path.join(gml_root, 'road.gml'))
            device_gml = self.create_citygml_cityfurniture(self.mesh_device)
            save_citygml(device_gml, os.path.join(gml_root, 'device.gml'))

        return self.mesh_road


class genBuilding:
    def __init__(self,
                 bdg_src_path,
                 bdg_obj_label_path='./data_src/src_3d/merged_filter1.csv',
                 bdg_obj_root='./data_src/src_3d/obj/',
                 probabilities=[1., 0., 0., 0., 0., 0., 0.],
                 low_storey=2,
                 high_storey=35,
                 **kwargs):

        self.bdg_src_path = bdg_src_path
        self.probabilities = probabilities
        self.bdg_src = gpd.read_file(self.bdg_src_path).geometry
        flattened_geometries = [geom for sublist in self.bdg_src.apply(flatten_multipolygons) for geom in sublist]
        self.bdg_src=gpd.array.GeometryArray(np.array(flattened_geometries))

        self.obj_mes = pd.read_csv(bdg_obj_label_path)
        self.obj_type = self.obj_mes['type'].values
        self.obj_type = self.type_map_bdg(self.obj_type)
        self.obj_root = bdg_obj_root

        self.low_storey = low_storey
        self.high_storey = high_storey

    # def crop_blg_poly(self, x_min, y_min, width=200., height=200.):
    #     self.roi_rect = box(x_min, y_min, x_min + width, y_min + height)
    #     self.roi_building = self.bdg_src[self.bdg_src.within(self.roi_rect)]
    #     return self.roi_building

    def type_map_bdg(self, data):
        data[(data == 5) | (data == 6) | (data == 7) | (data == 9) | (data == 12) | (data == 13)] = 5
        data[(data == 8) | (data == 11)] = 6
        data[(data == 10)] = 7
        return data

    def get_ab(self, polygon):
        rect = polygon.envelope
        rect = list(rect.exterior.coords)
        edge_lengths = [Point(rect[i]).distance(Point(rect[i - 1])) for i in range(1, len(rect))]

        return [max(edge_lengths), min(edge_lengths)]

    def get_reshape_scale(self, polygon1, polygon2):
        poly1_max, poly1_min = self.get_ab(polygon1)
        poly2_max, poly2_min = self.get_ab(polygon2)

        return min(poly1_max / poly2_max, poly1_min / poly2_min)

    def get_polygon(self, data):
        mesh_point = data.vertices
        min_z = np.min(mesh_point[:, 2])
        plane_origin = [0, 0, min_z + 1]
        plane_normal = [0, 0, 1]
        slice3d = data.section(plane_normal=plane_normal, plane_origin=plane_origin)
        if not slice3d:
            return

        slice2d, affn = slice3d.to_planar()
        slice_poly = slice2d.polygons_full
        if len(slice_poly) == 0:
            return
        slice_poly = np.column_stack((slice_poly[0].exterior.coords.xy))
        homo_array = np.hstack((slice_poly, np.zeros((len(slice_poly), 1)), np.ones((len(slice_poly), 1))))
        homo_3d = homo_array @ (affn.T)

        poly_xy = homo_3d[:, :2] / homo_3d[:, -1, None]

        return Polygon(poly_xy)

    def mesh_poly_iou(self, mesh, rot_shp_poly, center_poly, visualize=False):
        center_poly_xy = [center_poly.x, center_poly.y]
        tmp_mesh_poly = self.get_polygon(mesh)
        if not tmp_mesh_poly:
            return 0, 1
        tmp_mesh_b = calculate_bearing(tmp_mesh_poly)
        rot_mesh_poly = rotate(tmp_mesh_poly, tmp_mesh_b)
        center_mesh = rot_mesh_poly.centroid
        center_mesh_xy = [center_mesh.x, center_mesh.y]

        tmp_mesh_poly = translate(rot_mesh_poly, center_poly_xy[0] - center_mesh_xy[0],
                                  center_poly_xy[1] - center_mesh_xy[1])

        mesh_scale = self.get_reshape_scale(rot_shp_poly, tmp_mesh_poly)
        scaled_mesh_poly = scale(tmp_mesh_poly, xfact=mesh_scale, yfact=mesh_scale, origin=center_poly)

        iou = polygon_iou(rot_shp_poly, scaled_mesh_poly)

        if visualize:
            vis_polygon([rot_shp_poly, scaled_mesh_poly])

        return iou, mesh_scale

    def mesh_poly_transfer(self, mesh, polygon, mp_scale):
        mesh.apply_scale(mp_scale)
        mesh_point = mesh.vertices
        min_z = np.min(mesh_point[:, 2])

        tmp_mesh_poly = get_polygon(mesh)
        if not tmp_mesh_poly:
            return
        cen_tmp_mesh_poly = tmp_mesh_poly.centroid

        tmp_mesh_b = calculate_bearing(tmp_mesh_poly)
        shp_poly_b = calculate_bearing(polygon)
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.radians(tmp_mesh_b - shp_poly_b), [0, 0, 1], (cen_tmp_mesh_poly.x, cen_tmp_mesh_poly.y, 0))

        mesh.apply_transform(rotation_matrix)
        tmp_mesh_poly = get_polygon(mesh)
        cen_tmp_mesh_poly = tmp_mesh_poly.centroid
        cen_polygon = polygon.centroid
        mesh.apply_translation([cen_polygon.x - cen_tmp_mesh_poly.x, cen_polygon.y - cen_tmp_mesh_poly.y, -min_z])

        return mesh

    def gen_mesh_building_lod1(self, shp_polys, limit=None):
        self.mesh_building = []
        self.poly_building = []

        shp_l = len(shp_polys)
        for i in tqdm(range(shp_l)):
            shp_poly = shp_polys[i]
            if (limit is not None) and limit.intersection(shp_poly).any():
                continue
            self.poly_building.append(shp_poly)

            vertices, faces = polygon_to_mesh_3D(shp_poly)
            tmp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.mesh_building.append(tmp_mesh)
        self.building_limit = gpd.array.GeometryArray(np.array(self.poly_building))

        return self.mesh_building

    def gen_mesh_building_lod2(self, shp_polys, probabilities, limit=None, visualize=False):
        shp_l = len(shp_polys)

        # shp_polys = shp_poly.geometry.values
        self.mesh_building = []
        self.poly_building = []
        for i in tqdm(range(shp_l)):
            shp_poly = shp_polys[i]
            shp_poly_b = calculate_bearing(shp_poly)
            rot_shp_poly = rotate(shp_poly, shp_poly_b)
            center_poly = rot_shp_poly.centroid

            tmp_type_choice = random.choices(range(1, 8), probabilities)
            tmp_type_id = np.where(self.obj_type == tmp_type_choice[0])[0]
            idx_rand = random.sample(list(tmp_type_id), min(100, len(tmp_type_id)))

            mp_iou, mp_scale = 0, 1
            for j in idx_rand:
                tmp_mesh = trimesh.load(os.path.join(self.obj_root, self.obj_mes['id'].values[j] + '.obj'))
                tmp_mesh = tmp_mesh.dump(concatenate=True) if isinstance(tmp_mesh, trimesh.Scene) else tmp_mesh
                tmp_iou, tmp_scale = self.mesh_poly_iou(tmp_mesh, rot_shp_poly, center_poly)
                if tmp_iou > 0.85:
                    mp_iou, mp_scale = tmp_iou, tmp_scale
                    mesh_tar = tmp_mesh
                    break
                if tmp_iou > mp_iou:
                    mp_iou, mp_scale = tmp_iou, tmp_scale
                    mesh_tar = tmp_mesh

            mesh_tar = self.mesh_poly_transfer(mesh_tar, shp_poly, mp_scale)

            if mesh_tar:
                tmp_poly_building = get_polygon(mesh_tar)
                if (not limit) or (not np.array([i for i in limit.intersection(tmp_poly_building)]).any()):
                    self.poly_building.append(tmp_poly_building)
                    self.mesh_building.append(mesh_tar)

        self.building_limit = []
        for xx in self.mesh_building:
            self.building_limit.append(get_polygon(xx))
        if visualize:
            vis_polygon(self.building_limit)
        self.building_limit = gpd.array.GeometryArray(np.array(self.building_limit))

        return self.mesh_building

    def set_building_storey(self):
        low, high = self.low_storey * 3, self.high_storey * 3

        for tmp_mesh in self.mesh_building:
            rand_height = random.uniform(low, high)
            mesh_point = tmp_mesh.vertices
            mesh_point_z = mesh_point[:, 2]
            min_z = np.min(mesh_point_z)

            h_trans = rand_height - np.max(mesh_point_z)
            mesh_point[:, 2][mesh_point_z > (min_z + 1.)] += h_trans
            tmp_mesh.vertices = mesh_point

    def add_relief(self, points_relief):
        z_points_interpolate = relief_interpolate(self.mesh_building, points_relief)
        if (z_points_interpolate is None):
            return

        for i, tmp_mesh in enumerate(self.mesh_building):
            tmp_vertices = tmp_mesh.vertices
            tmp_vertices[:, 2] += np.min(z_points_interpolate[i])
            tmp_mesh.vertices = tmp_vertices

    def create_citygml_building(self, buildings, lod=1, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                                srsDimension="3"):
        nsmap = {
            'core': "http://www.opengis.net/citygml/2.0",
            'bldg': "http://www.opengis.net/citygml/building/2.0",
            'gml': "http://www.opengis.net/gml"
        }
        cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

        total_vertices = []
        for building in buildings:
            total_vertices.append(building.vertices)
        total_vertices = np.vstack(total_vertices)
        x_max, y_max, z_max = np.max(total_vertices, axis=0)
        x_min, y_min, z_min = np.min(total_vertices, axis=0)
        boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
        Envelope = etree.SubElement(boundedBy, "{http://www.opengis.net/gml}Envelope", srsName=srs_name,
                                    srsDimension=srsDimension)
        lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
        upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
        lowerCorner.text = '{} {} {}'.format(x_min, y_min, z_min)
        upperCorner.text = '{} {} {}'.format(x_max, y_max, z_max)

        if lod == 1:
            for building_data in buildings:
                vertices, faces = building_data.vertices, building_data.faces

                building_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
                building = etree.SubElement(building_member, "{http://www.opengis.net/citygml/building/2.0}Building")

                lod1Solid = etree.SubElement(building, "{http://www.opengis.net/citygml/building/2.0}lod1Solid")
                solid = etree.SubElement(lod1Solid, "{http://www.opengis.net/gml}Solid")
                exterior = etree.SubElement(solid, "{http://www.opengis.net/gml}exterior")
                compositeSurface = etree.SubElement(exterior, "{http://www.opengis.net/gml}CompositeSurface")

                for face in faces:
                    surfaceMember = etree.SubElement(compositeSurface, "{http://www.opengis.net/gml}surfaceMember")
                    polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                    exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                    linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                    posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                    coords = ' '.join(
                        ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                    coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                    posList.text = coords
        elif lod == 2:
            for building_data in buildings:
                vertices, faces = building_data.vertices, building_data.faces
                z_min, z_max = np.min(vertices[:, 2]), np.max(vertices[:, 2])

                building_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
                building = etree.SubElement(building_member, "{http://www.opengis.net/citygml/building/2.0}Building")

                measuredHeight = etree.SubElement(building,
                                                  "{http://www.opengis.net/citygml/building/2.0}measuredHeight")
                measuredHeight.text = str(round(z_max - z_min, 2))

                for face in faces:
                    boundedBy = etree.SubElement(building,
                                                 "{http://www.opengis.net/citygml/building/2.0}boundedBy")
                    z_face = vertices[face][:, 2]
                    if (z_face - z_min < 1.).all():
                        typeSurface = etree.SubElement(boundedBy,
                                                       "{http://www.opengis.net/citygml/building/2.0}GroundSurface")
                    elif (z_face - z_min > 1.).all():
                        typeSurface = etree.SubElement(boundedBy,
                                                       "{http://www.opengis.net/citygml/building/2.0}RoofSurface")
                    else:
                        typeSurface = etree.SubElement(boundedBy,
                                                       "{http://www.opengis.net/citygml/building/2.0}WallSurface")

                    lod2MultiSurface = etree.SubElement(typeSurface,
                                                        "{http://www.opengis.net/citygml/building/2.0}lod2MultiSurface")
                    MultiSurface = etree.SubElement(lod2MultiSurface, "{http://www.opengis.net/gml}MultiSurface")
                    surfaceMember = etree.SubElement(MultiSurface, "{http://www.opengis.net/gml}surfaceMember")
                    polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                    exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                    linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                    posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                    coords = ' '.join(
                        ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                    coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                    posList.text = coords

        return cityModel

    def gen_building_run(self, building_lod=2, limit=None, points_relief=None, visualize=False, save_gml=True,
                         gml_root=''):
        if building_lod == 1:
            self.gen_mesh_building_lod1(self.bdg_src, limit)
            self.set_building_storey()
        if building_lod == 2:
            self.gen_mesh_building_lod2(self.bdg_src, self.probabilities, limit, visualize)
            if self.low_storey and self.high_storey:
                self.set_building_storey()
        self.add_relief(points_relief)
        if save_gml:
            building_gml = self.create_citygml_building(self.mesh_building, building_lod)
            save_citygml(building_gml, os.path.join(gml_root, 'building.gml'))

        return self.mesh_building


class genVegetation:
    def __init__(self,
                 img_path,
                 vege_root='./data_src/src_3d/lod3veg/SolitaryVegetationObject/',
                 vege_label='./data_src/src_3d/tree_label.csv',
                 low_ratio=0.1,
                 high_ratio=1.,
                 **kwargs):
        check_img_ext(img_path)
        self.img_path = img_path
        self.vege_mes = pd.read_csv(vege_label)
        self.vege_id = self.vege_mes['id'].values
        self.vege_type = self.vege_mes['type'].values
        self.vege_root = vege_root
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio

    def gen_tree_mesh_lod1(self, limit_road, limit_bdg, dense=200):
        self.mesh_tree = []

        im_proj, im_Geotrans, im_data = read_tif(self.img_path)
        img_shape = im_data.shape
        if min(img_shape) == img_shape[2]:
            height, width, channel = im_data.shape
            im_data = im_data[:, :, :3]
        else:
            channel, height, width = im_data.shape
            im_data = np.transpose(im_data[:3, :, :], (1, 2, 0))

        im_data = im_data[:, :, ::-1]
        x_min, y_min, resolusion_x, resolusion_y = im_Geotrans[0], im_Geotrans[3], im_Geotrans[1], im_Geotrans[5]
        self.roi_rect = box(x_min, y_min, x_min + width * resolusion_x, y_min + height * resolusion_y)

        seg_contours = inference_image_vegetation(im_data, 'bg_extract/ckpt/yolov8_vegetation.pt')
        dst_poly = []
        for seg_contour in seg_contours:
            dst_poly.append(Polygon(seg_contour * [resolusion_x, resolusion_y] + [x_min, y_min]))
        dst_poly = gpd.array.GeometryArray(np.array(dst_poly))

        if limit_bdg is not None:
            limit_bdg = limit_bdg.buffer(3.)
        if limit_road is not None:
            limit_road = limit_road.buffer(3.)

        tar_xy = np.array([[random.uniform(x_min, x_min + width * resolusion_x) for _ in range(dense)],
                           [random.uniform(y_min, y_min + height * resolusion_y) for _ in range(dense)]]).T
        tmp_idx = []
        for i in range(len(tar_xy)):
            if limit_road is not None and Point(tar_xy[i]).within(limit_road).any():
                continue
            if limit_bdg is not None and Point(tar_xy[i]).within(limit_bdg).any():
                continue
            if Point(tar_xy[i]).within(dst_poly).any():
                tmp_idx.append(i)
        tar_xy = tar_xy[tmp_idx]

        high_num = int(len(tar_xy) * self.high_ratio / (self.high_ratio + self.low_ratio))

        for i in range(high_num):
            tree_poly = Point(tar_xy[i]).buffer(random.uniform(1., 3.))
            tree_height = random.uniform(6., 12.)

            vertices, faces = polygon_to_mesh_3D(tree_poly, tree_height)
            tmp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.mesh_tree.append(tmp_mesh)

        for i in range(high_num, len(tar_xy)):
            tree_poly = Point(tar_xy[i]).buffer(random.uniform(0.5, 2.))
            tree_height = random.uniform(2., 6.)

            vertices, faces = polygon_to_mesh_3D(tree_poly, tree_height)
            tmp_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            self.mesh_tree.append(tmp_mesh)

    def gen_tree_mesh_lod2(self, limit_bdg, limit_road, dense):
        self.mesh_tree = []

        im_proj, im_Geotrans, im_data = read_tif(self.img_path)
        img_shape=im_data.shape
        if min(img_shape)==img_shape[2]:
            height, width, channel = im_data.shape
            im_data=im_data[:,:,:3]
        else:
            channel, height, width = im_data.shape
            im_data = np.transpose(im_data[:3,:,:],(1,2,0))

        im_data=im_data[:,:,::-1]
        x_min,y_min,resolusion_x,resolusion_y=im_Geotrans[0],im_Geotrans[3],im_Geotrans[1],im_Geotrans[5]
        self.roi_rect = box(x_min, y_min, x_min + width * resolusion_x, y_min + height * resolusion_y)

        seg_contours = inference_image_vegetation(im_data, 'bg_extract/ckpt/yolov8_vegetation.pt')
        dst_poly = []
        for seg_contour in seg_contours:
            dst_poly.append(Polygon(seg_contour * [resolusion_x,resolusion_y]+[x_min,y_min]))
        dst_poly = gpd.array.GeometryArray(np.array(dst_poly))


        if limit_bdg is not None:
            limit_bdg = limit_bdg.buffer(3.)
        if limit_road is not None:
            limit_road = limit_road.buffer(3.)

        tar_xy = np.array([[random.uniform(x_min, x_min + width * resolusion_x) for _ in range(dense)],
                           [random.uniform(y_min, y_min + height * resolusion_y) for _ in range(dense)]]).T
        tmp_idx = []
        for i in range(len(tar_xy)):
            if limit_road is not None and Point(tar_xy[i]).within(limit_road).any():
                continue
            if limit_bdg is not None and Point(tar_xy[i]).within(limit_bdg).any():
                continue
            if Point(tar_xy[i]).within(dst_poly).any():
                tmp_idx.append(i)
        tar_xy = tar_xy[tmp_idx]

        high_num = int(len(tar_xy) * self.high_ratio / (self.high_ratio + self.low_ratio))
        low_num = int(len(tar_xy) * self.low_ratio / (self.high_ratio + self.low_ratio))
        high_idx = self.vege_id[self.vege_type == 1]
        low_idx = self.vege_id[self.vege_type == 0]
        high_idx_ = random.choices(list(range(len(high_idx))), k=high_num)
        low_idx_ = random.choices(list(range(len(low_idx))), k=low_num)

        for x, i in enumerate(high_idx_):
            tmp_mesh = trimesh.load(os.path.join(self.vege_root, high_idx[i] + '.obj'))
            tmp_mesh = tmp_mesh.dump(concatenate=True) if isinstance(tmp_mesh, trimesh.Scene) else tmp_mesh
            tmp_mesh_xy = tmp_mesh.centroid[:2]
            tmp_mesh_zmin = np.min(tmp_mesh.vertices[:, 2])

            tmp_trans = [tar_xy[x, 0] - tmp_mesh_xy[0], tar_xy[x, 1] - tmp_mesh_xy[1], -tmp_mesh_zmin]
            self.mesh_tree.append(tmp_mesh.apply_translation(tmp_trans))
        for x, i in enumerate(low_idx_):
            tmp_mesh = trimesh.load(os.path.join(self.vege_root, low_idx[i] + '.obj'))
            tmp_mesh = tmp_mesh.dump(concatenate=True) if isinstance(tmp_mesh, trimesh.Scene) else tmp_mesh
            tmp_mesh_xy = tmp_mesh.centroid[:2]
            tmp_mesh_zmin = np.min(tmp_mesh.vertices[:, 2])

            tmp_trans = [tar_xy[x + high_num, 0] - tmp_mesh_xy[0], tar_xy[x + high_num, 1] - tmp_mesh_xy[1],
                         -tmp_mesh_zmin]
            self.mesh_tree.append(tmp_mesh.apply_translation(tmp_trans))

    def add_relief(self, points_relief):
        z_points_interpolate = relief_interpolate(self.mesh_tree, points_relief)
        if (z_points_interpolate is None):
            return

        for i, tmp_mesh in enumerate(self.mesh_tree):
            tmp_vertices = tmp_mesh.vertices
            tmp_vertices[:, 2] += z_points_interpolate[i] + 0.01
            tmp_mesh.vertices = tmp_vertices

    def create_citygml_vegetation(self, vegetation, srs_name="http://www.opengis.net/def/crs/EPSG/0/30169",
                                  srsDimension="3"):
        nsmap = {
            'core': "http://www.opengis.net/citygml/2.0",
            'veg': "http://www.opengis.net/citygml/vegetation/2.0",
            'gml': "http://www.opengis.net/gml"
        }

        cityModel = etree.Element("{http://www.opengis.net/citygml/2.0}CityModel", nsmap=nsmap)

        total_vertices = []
        for tree in vegetation:
            total_vertices.append(tree.vertices)
        total_vertices = np.vstack(total_vertices)
        x_max, y_max, z_max = np.max(total_vertices, axis=0)
        x_min, y_min, z_min = np.min(total_vertices, axis=0)
        boundedBy = etree.SubElement(cityModel, "{http://www.opengis.net/gml}boundedBy")
        Envelope = etree.SubElement(boundedBy, "{http://www.opengis.net/gml}Envelope", srsName=srs_name,
                                    srsDimension=srsDimension)
        lowerCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}lowerCorner")
        upperCorner = etree.SubElement(Envelope, "{http://www.opengis.net/gml}upperCorner")
        lowerCorner.text = '{} {} {}'.format(x_min, y_min, z_min)
        upperCorner.text = '{} {} {}'.format(x_max, y_max, z_max)

        for vegetation_data in vegetation:
            vertices, faces = vegetation_data.vertices, vegetation_data.faces

            vegetation_member = etree.SubElement(cityModel, "{http://www.opengis.net/citygml/2.0}cityObjectMember")
            plantCover = etree.SubElement(vegetation_member,
                                          "{http://www.opengis.net/citygml/vegetation/2.0}SolitaryVegetationObject")

            lod2Geometry = etree.SubElement(plantCover,
                                            "{http://www.opengis.net/citygml/vegetation/2.0}lod2Geometry")
            multiSurface = etree.SubElement(lod2Geometry, "{http://www.opengis.net/gml}MultiSurface")

            for face in faces:
                surfaceMember = etree.SubElement(multiSurface, "{http://www.opengis.net/gml}surfaceMember")
                polygon = etree.SubElement(surfaceMember, "{http://www.opengis.net/gml}Polygon")
                exterior = etree.SubElement(polygon, "{http://www.opengis.net/gml}exterior")
                linearRing = etree.SubElement(exterior, "{http://www.opengis.net/gml}LinearRing")
                posList = etree.SubElement(linearRing, "{http://www.opengis.net/gml}posList")

                coords = ' '.join(
                    ['{} {} {}'.format(vertices[idx][0], vertices[idx][1], vertices[idx][2]) for idx in face])
                coords += ' {} {} {}'.format(vertices[face[0]][0], vertices[face[0]][1], vertices[face[0]][2])
                posList.text = coords

        return cityModel

    def gen_vege_run(self, limit_road, limit_bdg, points_relief=None,
                     dense=None,
                     lod=2,
                     save_gml=True, gml_root=''):
        if not dense:
            dense = random.randint(50, 200)
        if lod == 1:
            self.gen_tree_mesh_lod1(limit_road, limit_bdg, dense)
        elif lod == 2:
            self.gen_tree_mesh_lod2(limit_road, limit_bdg, dense)
        self.add_relief(points_relief)
        if save_gml and len(self.mesh_tree):
            vege_gml = self.create_citygml_vegetation(self.mesh_tree)
            save_citygml(vege_gml, os.path.join(gml_root, 'vegetation.gml'))

        return self.mesh_tree


def param_parser():
    parser = argparse.ArgumentParser(description="Parameters for real city generation. ")
    
    parser.add_argument('-i', '--input', help='Root path of building footprints. (GeoJSON)', type=str, nargs=1)
    parser.add_argument('--img', help='Input path of satellite image file. (GeoTIFF)', type=str, default='')
    parser.add_argument('--output_path', help='Root path of output file. ', type=str, nargs=1)
    
    parser.add_argument('-s', '--scale', help='Detail of the generated building. (0 - 2)', type=float, nargs=1, default=2.0)
    parser.add_argument('-r', '--random_seed', help='Global random seed of the generated scene. (0 - 65535)', type=int, default=1024)
    parser.add_argument('--crs', help='Set geo-reference for output file. (EPSG)', type=str, nargs=1, default='30169')

    parser.add_argument('--building_lod', help='LOD of generated building. (0 - 2) ', type=int, nargs=1, default=2)
    parser.add_argument('--building_storey', help='Storey range of buildings. (1 - 50) ', type=int, nargs=2, default=[1, 50])
    parser.add_argument('--building_type_ratio', help='Ratios of different roof types (Flat, Storey-difference Flat, Mixture, Slope1, Slope2, Pure Flat). \
                                                             Input the number in order. ', 
                        nargs=6, type=float, default=[0.2, 0.3, 0.3, 0.2, 0.0, 0.0])
    
    parser.add_argument('--road_lod', help='LOD of road objects. (0 - 2) ', type=int, nargs=1, default=1)
    parser.add_argument('--road_width', help='Range of road widths. (1 - 20) ', type=float, nargs=2, default=[1, 20])
    parser.add_argument('--road_width_ratio', help='Ratio of main road width and sidewalk width. (> 0.1) ', type=float, nargs=1, default=0.1)
    
    parser.add_argument('--veg_lod', help='LOD of vegetation objects. (0 - 2) ', type=int, nargs=1, default=2)
    parser.add_argument('--veg_height_ratio', help='Ratio of low vegetation and medium-high tree. (> 0.1) ', type=float, nargs=1, default=0.1)
    
    parser.add_argument('--device_lod', help='LOD of the cityFurniture. (0 - 2) ', type=int, nargs=1, default=2)
    parser.add_argument('--device_type_ratio', help='Ratio of utility poles and traffic lights. (> 0.1) ', type=float, nargs=1, default=0.1)
    
    parser.add_argument('--relief_lod', help='LOD of terrain surface. (0 - 1) ', type=int, nargs=1, default=1)
    
    args = parser.parse_args()
    return args
    

def main():
    params = param_parser()
    
    bldg_footprint = params.input
    satellite_image = params.img
    
    mesh_scale = params.scale
    random_seed = params.random_seed
    crs = params.crs
    
    gml_root_path = params.output_path
    geojson_path = './gen3d_realCity_testData/mapbox/test02/footprint/footprint_test_2_selected.geojson'
    obj_root_path = '../obj_geo'

    lod_building = params.building_lod
    storey_low, storey_high = params.building_storey
    bldg_type = params.building_type_ratio if lod_building == 2 else []
    prob_t1, prob_t2, prob_t3, prob_t4, prob_t5, prob_t6 = bldg_type
    
    lod_road = params.road_lod
    road_width_low, road_width_high = params.road_width
    road_width_sub = params.road_width_ratio
    
    lod_vegetation = params.veg_lod
    low_tree_ratio, high_tree_ratio = params.veg_height_ratio
    
    lod_device = params.device_lod
    telegraph_pole_ratio = params.device_type_ratio
    
    lod_relief = params.relief_lod
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # bg model
    img_path = satellite_image
    gen_road = genRoad(img_path=img_path)
    gen_road.crop_road_lineStr()
    mesh_road = gen_road.gen_road_run(road_lod=1, device_lod=2, gml_root=gml_root_path)
    road_limit = gen_road.road_limit

    # gen_building = genBuilding(bdg_src_path='/fast/zcb/data/PLATEAU_obj/gen3d_realCity/gen3d_realCity_testData/mapbox/test03/footprint/footprint_test_3_selected.geojson')
    # mesh_building = gen_building.gen_building_run(building_lod=2, limit=None, points_relief=None)

    gen_vegetation = genVegetation(img_path=img_path)
    mesh_vege = gen_vegetation.gen_vege_run(limit_road=None, limit_bdg=None, dense=2000, lod=2, gml_root=gml_root_path)

    res = mesh_vege + mesh_road
    combined_mesh = trimesh.util.concatenate(res)

    combined_mesh.export(os.path.join(obj_root_path, f'test_lod{lod_building}.obj'))

    # building
    polygons, names, origin_coords, pixel_sizes, footprint_images = read_geojson_and_rasterize(geojson_path)

    acc_vertices, acc_faces = [], []
    vertex_num = len(combined_mesh.vertices)
    # vertex_num = 0

    with open(os.path.join(obj_root_path, f'test_lod{lod_building}.obj'), 'a') as obj_file:
        obj_file.write('\n')

    if lod_building == 1:
        for idx, polygon in enumerate(polygons):
            img_Geotrans = np.array(
                [origin_coords[idx][0], pixel_sizes[idx][0], 0, origin_coords[idx][1], 0, pixel_sizes[idx][1]],
                dtype=np.float32)
            vertices, faces = bldg_lod1_gen_realCity(polygon[0], img_Geotrans, vertex_num)

            OBJ_output(os.path.join(obj_root_path, 'test_lod1.obj'), vertices, faces, vertex_num)

            vertex_num = vertex_num + len(vertices)
            acc_vertices.append(vertices)
            acc_faces.append(faces)

    elif lod_building == 2:
        model = create_model(cfg_file).cpu()
        index_list = [i for i in range(len(footprint_images))]
        random.shuffle(index_list)
        index_list_length = len(index_list)
        if not sum(bldg_type) == 1:
            print('Warning: Not full probabilities. Converting the remaining to Lod1-flat. ')

        ckpt_idx = 0
        ddim_sampler = ""
        for i, idx in enumerate(index_list):
            if not i % 10:
                model.load_state_dict(
                    load_state_dict(ckpt_files[ckpt_idx],
                                    location='cuda'))
                model = model.cuda()
                ddim_sampler = DDIMSampler(model)

                ckpt_idx += 1
                if ckpt_idx >= 5:
                    ckpt_idx = 0

            img_Geotrans = np.array(
                [origin_coords[idx][0], pixel_sizes[idx][0], 0, origin_coords[idx][1], 0, pixel_sizes[idx][1]],
                dtype=np.float32)
            vertices, faces = inference(model, ddim_sampler, footprint_images[idx], img_Geotrans, polygons[idx][0],
                                        vertex_num)

            OBJ_output(os.path.join(obj_root_path, 'test_lod2.obj'), vertices, faces, vertex_num, bldg_lod=2)

            acc_vertices.append(vertices)
            acc_faces.append([[coord + vertex_num for coord in face] for face in faces])
            vertex_num = vertex_num + len(vertices)

    save_citygml(bldg_citygml_realCity(acc_vertices, acc_faces, lod=lod_building, vertex_num=len(combined_mesh.vertices)), os.path.join(gml_root_path, 'building.gml'))


if __name__ == '__main__':
    main()

# vertex_num = 0
# obj_path = '/fast/zcb/data/PLATEAU_obj/gen3d_realCity/test_veg.obj'
# with open(obj_path, 'w') as obj_file:
#     pass

# for veg in mesh_vege:
    # for face in veg.faces:
    #     print(face)
# OBJ_output(obj_path, mesh_vege.vertices, mesh_vege.faces, vertex_num)
# vertex_num += len(mesh_vege.vertices) - 1
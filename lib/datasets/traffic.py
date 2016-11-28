# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from traffic_eval import traffic_eval
from fast_rcnn.config import cfg

class traffic(imdb):
    def __init__(self, image_set, data_path=None):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        self._data_path = self._get_default_path() if data_path is None \
                            else data_path
        #this is the classes of the intersection of training set(151) and testing set(136): 125
        self._classes = ('__background__', u'i1', u'i10', u'i11', u'i12', u'i13', u'i14', u'i2', u'i3', u'i4', u'i5', u'il100', u'il110', u'il50', u'il60', u'il70', u'il80', u'il90', u'io', u'ip', u'p1', u'p10', u'p11', u'p12', u'p13', u'p14', u'p15', u'p16', u'p17', u'p18', u'p19', u'p2', u'p22', u'p23', u'p25', u'p26', u'p27', u'p28', u'p3', u'p4', u'p5', u'p6', u'p8', u'p9', u'pa13', u'pa14', u'pb', u'pg', u'ph2', u'ph2.2', u'ph2.4', u'ph2.5', u'ph3', u'ph3.2', u'ph3.5', u'ph4', u'ph4.2', u'ph4.3', u'ph4.5', u'ph4.8', u'ph5', u'pl10', u'pl100', u'pl110', u'pl120', u'pl15', u'pl20', u'pl25', u'pl30', u'pl35', u'pl40', u'pl5', u'pl50', u'pl60', u'pl70', u'pl80', u'pl90', u'pm10', u'pm15', u'pm2', u'pm20', u'pm30', u'pm35', u'pm40', u'pm5', u'pm50', u'pm55', u'pm8', u'pn', u'pne', u'po', u'pr20', u'pr30', u'pr40', u'pr50', u'pr60', u'pr70', u'pr80', u'ps', u'pw3', u'pw3.2', u'pw3.5', u'pw4', u'w12', u'w13', u'w15', u'w16', u'w18', u'w20', u'w21', u'w22', u'w3', u'w30', u'w32', u'w34', u'w35', u'w42', u'w45', u'w46', u'w47', u'w55', u'w57', u'w58', u'w59', u'w63', u'wo')
        
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.png'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        #self._roidb_handler = self.selective_search_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup'     : True,
                       'use_salt'    : True,
                       'use_diff'    : False,
                       'matlab_eval' : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}

        assert os.path.exists(self._data_path), \
                'traffic data path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_index[i]

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /VOCdevkit2007/VOC2007/ImageSets/Main/val.txt
        image_set_file = os.path.join(self._data_path,'{}.txt'.format(self._image_set))
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where traffic is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'traffic')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def rpn_roidb(self):
        cache_file = os.path.join(self.cache_path,
                        self.name + '_roidb.pkl')

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} rpn roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = self.gt_roidb()
        #ss_roidb = self._load_selective_search_roidb(gt_roidb)
        #roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        roidb = gt_roidb

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote rpn roidb to {}'.format(cache_file)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print 'loading {}'.format(filename)
        assert os.path.exists(filename), \
               'rpn data not found at: {}'.format(filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_annotation(self, index):
        """
        Load image and bounding boxes info from txt file in the synthesis traffic file
        format.
        """
        filename = index.replace('images', 'labels').replace('png','txt').replace('jpg', 'txt')

        
        assert os.path.exists(filename), \
                'lable file does not exist: {}'.format(filename)
        # Load object bounding boxes into a data frame.
        with open(filename, 'r') as f:
            objs = [x.strip() for x in f.readlines()]
            
        num_objs = len(objs)    
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        for ix, obj in enumerate(objs):
            row = map(float,obj.split(' '))
            # Make pixel indexes 0-based
            x1 = row[1]
            y1 = row[2]
            x2 = row[3]
            y2 = row[4]
            cls = int(row[0])
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas} 

    def _get_comp_id(self):
        comp_id = (self._comp_id + '_' + self._salt if self.config['use_salt']
            else self._comp_id)
        return comp_id

    def _get_voc_results_file_template(self):
        # VOCdevkit/results/VOC2007/Main/<comp_id>_det_test_aeroplane.txt
        filename = self._get_comp_id() + '_det_' + self._image_set + '_{:s}.txt'
        dirpath = os.path.join(self._data_path,'results')
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        
        path = os.path.join(dirpath, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing class:{} results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def _do_python_eval(self, output_dir = 'output'):
        annopath = os.path.join(
            self._data_path,
            'labels',
            '{:s}.txt')
        imagesetfile = os.path.join(
            self._data_path,
            self._image_set + '.txt')
        cachedir = os.path.join(self._data_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric =  False
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = traffic_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.3,
                use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('Recompute with `./tools/reval.py --matlab ...` for your paper.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def _do_matlab_eval(self, output_dir='output'):
        print '-----------------------------------------------------'
        print 'Computing results with the official MATLAB eval code.'
        print '-----------------------------------------------------'
        path = os.path.join(cfg.ROOT_DIR, 'lib', 'datasets',
                            'VOCdevkit-matlab-wrapper')
        cmd = 'cd {} && '.format(path)
        cmd += '{:s} -nodisplay -nodesktop '.format(cfg.MATLAB)
        cmd += '-r "dbstop if error; '
        cmd += 'voc_eval(\'{:s}\',\'{:s}\',\'{:s}\',\'{:s}\'); quit;"' \
               .format(self._devkit_path, self._get_comp_id(),
                       self._image_set, output_dir)
        print('Running:\n{}'.format(cmd))
        status = subprocess.call(cmd, shell=True)

    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        self._do_python_eval(output_dir)
        #if self.config['matlab_eval']:
        #    self._do_matlab_eval(output_dir)
        if self.config['cleanup']:
            for cls in self._classes:
                if cls == '__background__':
                    continue
                filename = self._get_voc_results_file_template().format(cls)
                os.remove(filename)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True

if __name__ == '__main__':
    from datasets.traffic import traffic
    d = traffic('traffic')
    res = d.roidb
    from IPython import embed; embed()


import nibabel as nib  # for CT files reading
import numpy as np  # astandard numeric calculation library
import matplotlib.pyplot as plt  # for plotting
from skimage import measure, morphology, segmentation, feature, color  # morphological operations
import scipy.ndimage as ndimage
from scipy.ndimage.filters import convolve


class Processor(object):


    def SkeletonTHFinder(self, img):
        """
        same from last HW to use the skeleton in the ROI finding
        :param nifty_file: ct scanning
        :param plot: boolean if true, it plotss the number of connected components as a function of min threshold
        :return: Tuple(segmentation, minimum threshold, original segmentation, plot x axis, plot y axis)
        """

        max_th = 1300
        min_th = 350
        img = img.copy()
        img = self.SegmentationByTH(img, min_th, max_th)
        cc, cc_num = measure.label(img, return_num=True, connectivity=None)

        cc, cc_num = morphology.label(cc, return_num=True)
        labels_count = sorted(np.bincount(cc.flat)[1:], reverse=True)
        remove_small_objects_min_size = np.mean(labels_count) // 10
        remove_small_holes_area_th = 3 * 3 * 3 * 3 * 3
        backup = None
        cc_num_pre = -1
        for i in range(1):
            if labels_count[1] < labels_count[0] // 100:
                break
            backup = cc.copy()
            cc = morphology.closing(cc.astype(np.bool), selem=np.ones((20, 15, 5)))
            morphology.remove_small_objects(cc, min_size=remove_small_objects_min_size, connectivity=cc.ndim,
                                            in_place=True)
            cc = morphology.dilation(cc, selem=np.ones((5, 5, 2)))
            cc = morphology.closing(cc)
            morphology.remove_small_holes(cc, area_threshold=remove_small_holes_area_th, connectivity=cc.ndim,
                                          in_place=True)
            cc, cc_num = morphology.label(cc, return_num=True)
            labels_count = sorted(np.bincount(cc.flat)[1:], reverse=True)

            if 0 < cc_num <= 2:
                break
            elif cc_num < 1 or cc_num == cc_num_pre:
                cc = backup
                break
            cc_num_pre = cc_num

        for i in range(10):
            cc, cc_num = morphology.label(cc, return_num=True)
            cc_num_pre = cc_num
            backup = cc.copy()
            cc = morphology.erosion(cc, selem=np.ones((5, 5, 4)))
            cc, cc_num = morphology.label(cc, return_num=True)
            if cc_num > cc_num_pre or cc_num < 1:
                cc = backup
                break
        # enumerated back with indexes
        cc_binary = morphology.binary_closing(cc.astype(np.bool), selem=np.ones((15, 15, 1)))
        cc, cc_num = measure.label(cc_binary, return_num=True, connectivity=None)
        labels_count = sorted(list(enumerate(np.bincount(cc.flat)))[1:], key=lambda t: t[1], reverse=False)  # TODO
        remove_small_objects_min_size = labels_count[-1][1] // 100
        morphology.remove_small_objects(cc, min_size=remove_small_objects_min_size, connectivity=cc.ndim,
                                        in_place=True)
        constructed = cc == labels_count[-1][0]
        constructed = np.bool_(constructed).astype(np.int16)
        return constructed

    def SegmentationByTH(self, nifty_file, Imin, Imax):
        """
        Segmenting the CT scan in the nifty file according to the given threshold
        :param nifty_file: name of the CT scan file
        :param Imin: minimum intensity threshold
        :param Imax: maximum intensity threshold
        :return: returns a numpy segmented image
        """
        if isinstance(nifty_file, str):
            nifty_file = nib.load(f"./data/{nifty_file}")
        if not isinstance(nifty_file, np.ndarray):
            img_data = nifty_file.get_fdata()
        else:
            img_data = nifty_file
        img_data_th = np.zeros_like(img_data)
        indexes = (Imin < img_data) & (img_data < Imax)
        img_data_th[indexes] = 1

        return img_data_th.astype(np.bool)

    def IsolateBody(self, ct):
        """
        thresholds the ct for -500 to 2000 HU, removes small objects then find largest CC and return
        :param ct: np array
        :return: isolated body in np array
        """
        th_ct = self.SegmentationByTH(ct.copy(), -500, 2000)
        filtered = morphology.remove_small_objects(th_ct)
        labeled, l_num = morphology.label(filtered, return_num=True)
        largest_label = np.argmax(np.bincount(labeled.flat)[1:]) + 1
        return np.asarray(labeled == largest_label).astype(np.int16)  # body segmentation

    def IsolateBS(self, body_segmentation):
        """
        Isolate lungs, CC and BB cuts
        :param body_segmentation:
        :return: the sedmentations and the two cuts
        """
        body_segmentation_upper = body_segmentation.copy()
        body_segmentation_upper[:, :, :int(0.45*body_segmentation.shape[2])] = 0
        selem = np.ones((3, 3, 3))
        body_segmentation_upper = morphology.closing(body_segmentation_upper, selem=selem)
        lungs = np.zeros_like(body_segmentation_upper)
        for z in range(body_segmentation_upper.shape[2]):
            mask = ndimage.binary_fill_holes(body_segmentation_upper[:, :, z]).astype(np.int16)
            lungs[:, :, z] = body_segmentation_upper[:, :, z] - mask

        lungs_labels = measure.label(lungs)
        bin_count = np.bincount(lungs_labels.flat)[1:]
        lung_lbl = np.argmax(bin_count) + 1
        lungs = np.asarray(lungs_labels == lung_lbl)

        _, _, z_none_zero = np.nonzero(lungs)
        bin_cc = np.bincount(z_none_zero)

        # give larger weights to the edges of the lung tissues (e.g lung 5 4 3 2 1 0 1 2 3 4 5 lung)
        weight_vec_cc_y = np.abs(np.arange(-lungs.shape[1] // 2, int(np.ceil(lungs.shape[1] / 2))))
        weight_vec_cc_x = np.abs(np.arange(-lungs.shape[0] // 2, int(np.ceil(lungs.shape[0] / 2))))
        weight_mat_cc_y = np.zeros((lungs.shape[0], lungs.shape[1]))
        weight_mat_cc_x = np.zeros((lungs.shape[1], lungs.shape[0]))
        weight_mat_cc_y[:] = weight_vec_cc_y
        weight_mat_cc_x[:] = weight_vec_cc_x
        weight_mat_cc_x = weight_mat_cc_x.T
        weight_mat_cc = weight_mat_cc_x + weight_mat_cc_y
        weight_mat_cc_x = weight_mat_cc_y = weight_vec_cc_y = weight_vec_cc_x = None  # release to GC

        cc_upper_bound = np.argmax(bin_cc)
        bb = np.min(z_none_zero)
        # try to find thicker slice by checking score of the lower slices, and get max
        cc_score_max = np.sum(weight_mat_cc * lungs[:, :, cc_upper_bound])
        cc = cc_upper_bound
        while cc_upper_bound > bb:
            cc_upper_bound -= 1
            cc_score = np.sum(weight_mat_cc * lungs[:, :, cc_upper_bound])
            if cc_score > cc_score_max:
                cc_score_max = cc_score
                cc = cc_upper_bound

        return lungs.astype(np.int16), cc, bb

    def ThreeDBand(self, body, lungs, cc, bb):
        """
        calculate the three band by nullifying above cc and below bb
        :param body: body segmentation
        :param lungs: lungs segmentation
        :param cc: cc cut index in z-axis
        :param bb: bb cut index in z-axis
        :return:
        """
        body_bb_cc = body.copy()
        lungs_bb_cc = lungs.copy()
        lungs_bb_cc_chull = None
        for z in range(cc+1, bb, -1):
            if z > int((cc-bb)//5 + bb):
                lungs_bb_cc_chull = morphology.convex_hull_image((lungs_bb_cc[:, :, z]))
            if lungs_bb_cc_chull is not None and np.max(lungs_bb_cc_chull) > 0:
                body_bb_cc[lungs_bb_cc_chull, z] = 0

        body_bb_cc[:, :, :bb] = 0
        lungs_bb_cc[:, :, :bb] = 0
        body_bb_cc[:, :, cc:] = 0
        lungs_bb_cc[:, :, cc:] = 0
        body_bb_cc[:, :, cc] = body_bb_cc[:, :, bb] = 0
        return body_bb_cc

    def mergedROI(self, full_img, aorta_seg):
        """
        merge between the convex hull and a stacked rectangle around the vertebra
        :param full_img: image np array
        :param aorta_seg: aorta segmentation as mask np array
        :return: returns the merged mask in one np array
        """
        body_seg = self.IsolateBody(full_img)
        lungs_seg, cc, bb = self.IsolateBS(body_seg)
        chull_seg = self.ThreeDBand(body_seg, lungs_seg, cc, bb)
        bone_seg = self.SegmentationByTH(full_img, 300, 1300)
        x_aorta, y_aorta, z_aorta = np.nonzero(aorta_seg[:, :, :bb])
        u_z_aorta = np.sort(np.unique(z_aorta))
        u_y_aorta = np.sort(np.unique(y_aorta))
        half_aorta = u_z_aorta[len(u_z_aorta)//2]
        bone_seg[:, :np.min(u_y_aorta), :] = 0
        bone_seg = morphology.dilation(bone_seg, selem=np.ones((5, 10, 1)))
        mask = np.zeros_like(bone_seg)
        x_min_acc = 0
        y_min_acc = 0
        x_max_acc = 0
        y_max_acc = 0
        c = 0
        for z in range(np.min(u_z_aorta), cc):

            labeled = morphology.label(bone_seg[:, :, z], connectivity=2)
            b_count = np.bincount(labeled.flat)[1:]
            argmax = np.argmax(b_count)
            largest_label = argmax + 1
            b_count[argmax] = -1
            s_argmax = np.argmax(b_count)
            s_largest_label = s_argmax + 1

            labeled[:labeled.shape[0]//3, :] = 0
            labeled[2*labeled.shape[0]//3:, :] = 0
            vertebra = np.logical_or(labeled == largest_label, labeled == s_largest_label).astype(np.int16)  # body segmentation
            slice_x, slice_y = ndimage.find_objects(vertebra)[0]
            c += 1
            x_min_acc += slice_x.start
            x_max_acc += slice_x.stop
            y_min_acc += slice_y.start
            y_max_acc += slice_y.stop
        # calculate avg coordinates of bounding rect
        avg_min_x = x_min_acc // max(c, 1)
        avg_min_y = y_min_acc // max(c, 1)
        avg_max_x = x_max_acc // max(c, 1)
        avg_max_y = y_max_acc // max(c, 1)
        mask[avg_min_x:avg_max_x+1, avg_min_y:avg_max_y+1, np.min(u_z_aorta):cc] = 1

        return mask + chull_seg

    def _chull(self, img, start, stop, keep_like=-1):
        """
        helper function for calculating convex hulls
        :param img: np array
        :param start: star range in z-axis
        :param stop: end range in z-axis
        :param keep_like: stack with the keep_like convex hull till the end of range
        :return: stacked convex hull images
        """
        mask = np.zeros_like(img)
        chull = None
        for z in range(start, stop, -1):
            if z > keep_like:
                chull = morphology.convex_hull_image(img[:, :, z])
            mask[:, :, z] = chull
        return mask

    def liverROI(self, full_img, aorta_seg):
        """
        depending on thresholding, isolate body and merged ROI the function creates liver ROI.
        :param full_img: np array of full ct
        :param aorta_seg: aorta segmentation mask
        :return: liver ROI as np array
        """
        roi = full_img.copy()
        roi = self.SegmentationByTH(roi, -100, 200)
        body_seg = self.IsolateBody(full_img)
        lungs_seg, cc, bb = self.IsolateBS(body_seg)
        x_aorta, _, z_aorta = np.nonzero(aorta_seg)
        aorta_edge = np.min(x_aorta)
        roi[:, :, int((cc-bb)//2 + bb):] = 0
        roi[:, :, :int(0.7*bb)] = 0
        roi = roi * np.logical_not(lungs_seg)
        roi[:int(0.8*aorta_edge), :, :] = 0
        roi = roi * body_seg
        merged_roi = self.mergedROI(full_img, aorta_seg)
        roi = roi * np.logical_not(merged_roi)
        seg_skeleton = self.SkeletonTHFinder(full_img)
        seg_skeleton_dilated = morphology.dilation(seg_skeleton, selem=np.ones((3, 3, 30)))
        roi = roi * np.logical_not(seg_skeleton_dilated)
        seg_skeleton_erosed = morphology.dilation(seg_skeleton_dilated, selem=np.ones((3, 3, 1)))
        z_aorta = np.unique(z_aorta)
        chull_skel = self._chull(seg_skeleton_erosed, np.max(z_aorta), np.min(z_aorta), (np.max(z_aorta)+np.min(z_aorta))//2)
        roi = chull_skel * roi
        return roi.astype(np.int16)

    def findSeeds(self, roi):
        """
        the seeds are sampled in two phases one for each liver lobe. in the right lobe there's more freedom in Y-Z axes
        while in left lobe its more restricted. both are sampled from none-zero liver roi mask voxels.
        :param roi: liver roi mask
        :return: tuple(x-array, y-array, z-array)
        """
        x, y, z = np.nonzero(roi)
        zipped = zip(x, y, z)
        max_z = np.max(z)
        mid_z = int((max_z - np.min(z)) // 2 + np.min(z))
        mid_x = int(3*(np.max(x) - np.min(x)) // 4 + np.min(x))
        mid_y = int((np.max(y) - np.min(y)) // 6 + np.min(y))

        indexes = np.arange(x.shape[0])

        # seed for the right lobe (the larger part)
        right_lobe_indices = indexes[int(len(indexes)*0.75):int(len(indexes)*0.95)]
        right_lobe_indices = np.random.choice(right_lobe_indices, size=220, replace=False)

        # seeds for the liver left lobe (the smaller part)
        left_lobe = filter(lambda t: t[0] < mid_x and t[1] < mid_y and mid_z <= t[2] < 1.05*mid_z, zipped)
        left_lobe_x, left_lobe_y, left_lobe_z = [], [], []
        appendix_len = 0
        for _ in left_lobe:
            left_lobe_x.append(_[0])
            left_lobe_y.append(_[1])
            left_lobe_z.append(_[2])
            appendix_len += 1
        left_lobe_ind = np.random.choice(range(appendix_len), size=40, replace=False)
        left_lobe_x, left_lobe_y, left_lobe_z = np.array(left_lobe_x), np.array(left_lobe_y), np.array(left_lobe_z)
        left_lobe_x, left_lobe_y, left_lobe_z = left_lobe_x[left_lobe_ind], left_lobe_y[left_lobe_ind], left_lobe_z[left_lobe_ind]

        x = np.concatenate((x[right_lobe_indices], left_lobe_x))
        y = np.concatenate((y[right_lobe_indices], left_lobe_y))
        z = np.concatenate((z[right_lobe_indices], left_lobe_z))
        return x, y, z

    def bfs(self, img_3d, seeds, roi):
        """
        The main algo for SRG, I depend mainly on morphological operations:
        I set the seeds in a zero matrix as 1-s then label the CCs.
        none_zero <--- hold counter for non zero in the liver roi mask.
        while explored_voxels < none_zero:
            dilate explored_voxels to 26 neighbors of every voxel
            check which of them satisfy the criteria (np.abs((INTENSITY_VOXEL(AVERAGED) - REGION_MEAN_INTENSITY) / SD))
            add the accepted ones to the "labeled" matrix
            re-lable CCs in "labeled" and "dilated"
        post-process with dilation and closing.
        :param img_3d: np 3d matrix = main ct * roi
        :param seeds: tuple(x-array, y-array, z-array)
        :param roi: liver ROI mask
        :return: segmented liver mask np array
        """
        def sd_intensity(regionmask, intensity_image):
            """ to be used as extra property in measure.regionprops """
            return np.std(intensity_image)

        none_zero = np.count_nonzero(roi)
        threshold = 0.8
        labeled = np.zeros_like(img_3d)
        labeled[seeds[0], seeds[1], seeds[2]] = 1
        labeled = morphology.label(labeled).astype(np.int16)
        dilated = labeled.copy()
        none_zero_counter = 0
        c = 0
        kernel = np.array([[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]) / 14
        while none_zero_counter < none_zero:
            none_zero_counter = np.count_nonzero(dilated)
            print(f'------{c}-----------')
            c += 1
            means_ = []
            stds_ = []
            for prop in measure.regionprops(labeled, intensity_image=img_3d, extra_properties=(sd_intensity,)):
                means_.append(prop.mean_intensity)
                sd = prop.sd_intensity
                stds_.append(sd if sd != 0 else 1)
            means = np.array([0]+means_)
            stds = np.array([1]+stds_)
            pre_dilation = dilated.copy()
            morphology.dilation(dilated, selem=np.ones((3, 3, 3)), out=dilated)
            diff_lbl_means = dilated - pre_dilation
            diff_lbl_stds = diff_lbl_means.copy()
            diff_lbl_means.flat = means[diff_lbl_means.flat]
            diff_lbl_stds.flat = stds[diff_lbl_stds.flat]
            diff_intensity = img_3d * np.bool_(diff_lbl_means)
            diff_intensity = convolve(diff_intensity, kernel)  # for smoothing intensities around checked voxel
            diff_lbl_stds[diff_lbl_stds == 0] = 1  # prevent div by 0
            tmp = (np.abs((diff_intensity - diff_lbl_means) / diff_lbl_stds))  # similarity criteria
            accepted_voxels_mask = tmp < threshold
            labeled = accepted_voxels_mask * diff_lbl_means + labeled
            labeled, labels_num = morphology.label(np.bool_(labeled), return_num=True)
            labeled = labeled.astype(np.int16)
            dilated, labels_num_d = morphology.label(np.bool_(dilated), return_num=True)
            dilated = dilated.astype(np.int16)
        # post-process
        labeled = morphology.binary_dilation(labeled, selem=np.ones((10,10,10)))
        morphology.remove_small_holes(labeled, connectivity=labeled.ndim, area_threshold=128, in_place=True)
        labeled = morphology.binary_closing(labeled, selem=np.ones((3,3,3)))
        morphology.remove_small_objects(labeled, connectivity=labeled.ndim, min_size=10, in_place=True)
        return np.bool_(labeled).astype(np.int16)

    def multipleSeedsRG(self, ct, roi):
        """
        multiple seeds sampled then passed to a bfs based algorithm to region grow
        :param ct: ct np array
        :param roi: liver ROI np array
        :return: liver segmentation
        """
        ct_roi = ct * roi
        seeds = self.findSeeds(roi)
        liver_seg = self.bfs(ct_roi, seeds, roi)
        self.sample_stack(liver_seg, axis=2, start_with=160, show_every=5, title='liver_seg')
        self.sample_stack(ct * roi, axis=2, start_with=160, show_every=5, title='ct * roi')
        return liver_seg

    def evaluateSegmentation(self, GT_seg, est_seg):
        """
        :param GT_seg: ground truth segmentation
        :param est_seg: estimated segmentation
        :return: Overlap difference and DICE coefficient
        """
        dice = 2 * (np.sum(GT_seg * est_seg)) / max((np.sum(GT_seg) + np.sum(est_seg)), 1)
        vod = 1 - np.sum(GT_seg * est_seg) / max(np.sum(np.bool_(GT_seg + est_seg)), 1)
        return dice, vod

    def segmentLiver(self, ct_name, aorta_name, output_name):
        """
        :param ct_name:
        :param aorta_name:
        :param output_name:
        :return: open the files, rotate if hard ct. find liver roi. sample seeds. then segemtn liver and return.
        """
        is_hard = 'hard' in ct_name.lower()
        nifti_file_1 = nib.load(ct_name)
        ct = nifti_file_1.get_fdata()
        nifti_file_2 = nib.load(aorta_name)
        aorta = nifti_file_2.get_fdata()
        if is_hard:
            aorta = np.rot90(np.rot90(aorta))
            ct = np.rot90(np.rot90(ct))
        roi = self.liverROI(ct, aorta)
        liver_seg = self.multipleSeedsRG(ct, roi)
        liver_seg_n = nib.Nifti1Image(liver_seg, nifti_file_1.affine)
        nib.save(liver_seg_n, output_name)
        return liver_seg

    def sample_stack(self, stack, rows=6, cols=6, start_with=10, show_every=3, axis=2, title='No Name', cm='gray'):
        """
        auxiliary function to display sampled slices of 3d np image
        """
        fig, ax = plt.subplots(rows, cols, figsize=[30, 30])
        fig.suptitle(title)
        for i in range(rows * cols):
            ind = start_with + i * show_every
            ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
            if axis == 2 and len(stack.shape) > 2 and ind < stack.shape[2]:
                ax[int(i / rows), int(i % rows)].imshow(stack[:, :, ind], cmap=cm)
            elif axis == 1 and ind < stack.shape[1]:
                ax[int(i / rows), int(i % rows)].imshow(stack[:, ind, :], cmap=cm)
            elif axis == 0 and ind < stack.shape[0]:
                ax[int(i / rows), int(i % rows)].imshow(stack[ind, :, :], cmap=cm)
            ax[int(i / rows), int(i % rows)].axis('off')
        plt.show()


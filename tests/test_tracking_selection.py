import unittest

from mosaichunk_h3.tracking_selection import automatic_regions
from mosaichunk_h3.region_memory import RegionPlan,combine_plans


class AutomaticSelectionTests(unittest.TestCase):
    def candidate(self,box,area,quality=.99):
        return dict(bbox=box,area=area,predicted_iou=quality,stability_score=.99,segmentation='mask')

    def test_automatic_selection_rejects_background_low_confidence_and_duplicates(self):
        masks=[self.candidate((0,0,100,100),10000),
               self.candidate((10,10,30,50),1500),
               self.candidate((11,10,30,50),1500),
               self.candidate((60,30,20,40),800),
               self.candidate((10,70,20,20),400,.1)]
        regions=automatic_regions(masks,100,100)
        self.assertEqual(len(regions),2)
        self.assertEqual(regions[0]['name'],'auto object 1')
        self.assertTrue(all(0<=v<=1 for r in regions for v in r['box']))

    def test_no_confident_objects_is_valid(self):
        self.assertEqual(automatic_regions([],100,100),[])

    def test_global_anchor_cannot_become_a_regional_target(self):
        global_plan=RegionPlan(1,3,'whole frame',(0,0,1,1),(0,0,1,1),1.,1,3,regional=False)
        conflicting=RegionPlan(0,1,'region',(0,0,1,1),(0,0,1,1),1.,0,1)
        self.assertEqual(combine_plans((global_plan,),(conflicting,)),(global_plan,))


if __name__=='__main__':unittest.main()

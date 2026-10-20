"""Bounded SAM mask selection; no imports, downloads or model loading here."""
import math


def automatic_regions(masks, width, height, *, limit=4):
    """Choose diverse confident object-sized masks, never a whole-image prompt."""
    proposals = []
    for item in masks:
        quality = float(item.get('predicted_iou',0.)) * float(item.get('stability_score',0.))
        area = float(item.get('area',0.)) / max(1,width*height)
        if quality < .65 or not .01 <= area <= .8:
            continue
        x,y,w,h = (float(v) for v in item['bbox'])
        box = (max(0.,x/width),max(0.,y/height),min(1.,(x+w)/width),min(1.,(y+h)/height))
        if not box[0] < box[2] or not box[1] < box[3]:
            continue
        if (box[2]-box[0])*(box[3]-box[1]) > .95:
            continue
        proposals.append((quality*math.sqrt(area),box,item['segmentation']))
    proposals.sort(key=lambda entry: entry[0],reverse=True)
    selected = []
    for _,box,mask in proposals:
        duplicate = False
        area = (box[2]-box[0])*(box[3]-box[1])
        for prior in selected:
            b = prior['box']
            overlap = max(0.,min(box[2],b[2])-max(box[0],b[0])) * max(0.,min(box[3],b[3])-max(box[1],b[1]))
            union = area+(b[2]-b[0])*(b[3]-b[1])-overlap
            if overlap/max(union,1e-8) > .65:
                duplicate = True
                break
        if not duplicate:
            selected.append(dict(name=f'auto object {len(selected)+1}',box=box,segmentation=mask))
        if len(selected) == limit:
            break
    return selected

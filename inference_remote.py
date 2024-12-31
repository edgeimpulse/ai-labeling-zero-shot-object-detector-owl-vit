import argparse, math, os, sys, io
import time, json
import edgeimpulse_api as ei
import requests
import base64
import numpy as np
from PIL import Image

if not os.getenv('EI_PROJECT_API_KEY'):
    print('Missing EI_PROJECT_API_KEY')
    sys.exit(1)
if not os.getenv('BEAM_ENDPOINT'):
    print('Missing BEAM_ENDPOINT')
    sys.exit(1)
if not os.getenv('BEAM_ACCESS_KEY'):
    print('Missing BEAM_ACCESS_KEY')
    sys.exit(1)

EI_API_KEY = os.environ.get("EI_PROJECT_API_KEY")
EI_API_ENDPOINT = os.environ.get("EI_API_ENDPOINT", "https://studio.edgeimpulse.com/v1")
BEAM_ENDPOINT = os.environ.get("BEAM_ENDPOINT", '')
BEAM_ACCESS_KEY = os.environ.get("BEAM_ACCESS_KEY", '')

parser = argparse.ArgumentParser(description="Zero-shot object detector (running in Beam.cloud)")
parser.add_argument("--prompt", type=str, required=True,
    help='Items to detect, split by newlines. ' +
        'Desired class name and min. confidence rating in parenthesis. E.g. "beer bottle (beer, 0.2)"')
parser.add_argument("--data-ids-file", type=str, required=True,
    help='File with IDs (as JSON)')
parser.add_argument("--propose-actions", type=int, required=False,
    help='If this flag is passed in, only propose suggested actions')
parser.add_argument("--delete_existing_bounding_boxes", type=str, required=True,
    help='What to do with existing bounding boxes (either "no", "matching-prompt" or "yes")')
parser.add_argument("--nms", action='store_true',
    help='Runs non-max suppression if passed in')
parser.add_argument("--nms-iou-threshold", required=False, type=float, default=0.2,
    help='IOU threshold for NMS')
parser.add_argument("--ignore-objects-smaller-than", required=False, type=float,
    help='If specified, ignores objects smaller than X%')
parser.add_argument("--ignore-objects-larger-than", required=False, type=float,
    help='If specified, ignores objects larger than X%')

args, unknown = parser.parse_known_args()

configuration = ei.Configuration(host=EI_API_ENDPOINT)
configuration.api_key["ApiKeyAuthentication"] = EI_API_KEY
api = ei.ApiClient(configuration)
projects_api = ei.ProjectsApi(api)
raw_data_api = ei.RawDataApi(api)

EI_PROJECT_ID = projects_api.list_projects().projects[0].id

def box_iou_batch(
	boxes_a: np.ndarray, boxes_b: np.ndarray
) -> np.ndarray:

    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area_a = box_area(boxes_a.T)
    area_b = box_area(boxes_b.T)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[:, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[:, 2:])

    area_inter = np.prod(
    	np.clip(bottom_right - top_left, a_min=0, a_max=None), 2)

    return area_inter / (area_a[:, None] + area_b - area_inter)

def non_max_suppression(
   predictions: np.ndarray, iou_threshold: float = 0.5
) -> np.ndarray:
    rows, columns = predictions.shape

    sort_index = np.flip(predictions[:, 4].argsort())
    predictions = predictions[sort_index]

    boxes = predictions[:, :4]
    categories = predictions[:, 5]
    ious = box_iou_batch(boxes, boxes)
    ious = ious - np.eye(rows)

    keep = np.ones(rows, dtype=bool)

    for index, (iou, category) in enumerate(zip(ious, categories)):
        if not keep[index]:
            continue

        condition = (iou > iou_threshold) & (categories == category)
        keep = keep & ~condition

    return keep[sort_index.argsort()]

# the replacement looks weird; but if calling this from CLI like "--prompt 'test\nanother line'" we'll get this still escaped
# (you could use $'test\nanotherline' but we won't do that in the Edge Impulse backend)
prompt = args.prompt.replace('\\n', '\n')
delete_existing = args.delete_existing_bounding_boxes
if delete_existing != 'no' and delete_existing != 'matching-prompt' and delete_existing != 'yes':
    print('Invalid value for --delete_existing_bounding_boxes')
    exit(1)

if args.data_ids_file:
    with open(args.data_ids_file, 'r') as f:
        data_ids = json.load(f)

objects = []

for line in prompt.splitlines():
    if line == '': continue
    parenthesis_start = line.rfind('(')
    parenthesis_end = line.rfind(')')
    if parenthesis_start != -1 and parenthesis_end != -1:
        search_for = line[0:parenthesis_start].strip()
        parenthesis = line[parenthesis_start+1:parenthesis_end]
        if (',' in parenthesis):
            label = parenthesis.split(',')[0].strip()
            min_confidence = float(parenthesis.split(',')[1].strip())
        else:
            label = parenthesis.strip()
            min_confidence = 0.0
    else:
        search_for = line.strip()
        label = search_for
        min_confidence = 0.0

    objects.append({
        'search_for': search_for,
        'label': label,
        'min_confidence': min_confidence
    })

print('Labeling data using zero-shot object detection w/ google/owlv2-base-patch16-ensemble')
print('')
print('Detecting objects:')
print('    Prompts:')
for object in objects:
    print('        - ' + object['search_for'] + ', label: ' + object['label'] + ', min confidence: ' + str(object['min_confidence']))

if (len(data_ids) < 6):
    print('    IDs:', ', '.join([ str(x) for x in data_ids ]))
else:
    print('    IDs:', ', '.join([ str(x) for x in data_ids[0:5] ]), 'and ' + str(len(data_ids) - 5) + ' others')

print('')

print('Note: it might take a minute to spin up a new GPU')
print('')

def current_ms():
    return round(time.time() * 1000)

ix = 0
for data_id in data_ids:
    ix = ix + 1
    now = current_ms()

    sample = (raw_data_api.get_sample(project_id=EI_PROJECT_ID, sample_id=data_id, proposed_actions_job_id=args.propose_actions)).sample
    prefix = '[' + str(ix).rjust(len(str(len(data_ids))), ' ') + '/' + str(len(data_ids)) + ']'

    print(prefix, 'Labeling ' + sample.filename + ' (ID ' + str(sample.id) + ')...', end='')

    new_metadata = sample.metadata if sample.metadata else { }
    new_metadata['labeled_by'] = 'owlv2'
    new_metadata['prompt'] = prompt

    res = requests.get(url=EI_API_ENDPOINT + '/api/' + str(EI_PROJECT_ID) + '/raw-data/' + str(data_id) + '/image',
        headers={
            'x-api-key': EI_API_KEY,
        },
    )
    if (res.status_code != 200):
        raise Exception('Failed to fetch sample from Edge Impulse (status_code=' + str(res.status_code) + '): ' + res.content.decode("utf-8"))

    image = Image.open(io.BytesIO(res.content)).convert("RGB")
    image_width, image_height = image.size
    image_area = image_width * image_height

    body = json.dumps({
        'base64_image': base64.b64encode(res.content).decode('utf-8'),
        'labels': [x['search_for'] for x in objects],
    })

    print(' Running inference...', end='')

    res = requests.post(url=BEAM_ENDPOINT,
        headers={
            'Authorization': 'Bearer ' + BEAM_ACCESS_KEY,
            'Content-Type': 'application/json',
        },
        data=body
    )
    if (res.status_code != 200):
        raise Exception('Failed to classify sample (status_code=' + str(res.status_code) + '): ' + res.content.decode("utf-8"))

    # print('res.content', res.content)

    predictions = json.loads(res.content.decode('utf-8'))['predictions']
    # print('predictions', predictions)

    prediction_time = current_ms() - now

    objects_by_search_for = {}
    labels_in_prompt = []
    for o in objects:
        objects_by_search_for[o['search_for']] = o
        labels_in_prompt.append(o['label'])

    bbs = []
    if delete_existing == 'no':
        bbs = sample.bounding_boxes
    elif delete_existing == 'matching-prompt':
        for bb in sample.bounding_boxes:
            if (not bb.label in labels_in_prompt):
                bbs.append(bb)
    elif delete_existing == 'yes':
        bbs = [] # <-- clear out all bounding boxes

    prediction_per_label = {}

    for prediction in predictions:
        object = objects_by_search_for[prediction['label']]

        box = prediction["box"]
        label = prediction["label"]
        score = prediction["score"]

        if (score < object['min_confidence']): continue

        xmin, ymin, xmax, ymax = box.values()

        area_px = (xmax - xmin) * (ymax - ymin)
        bb_area = (area_px / image_area) * 100

        if args.ignore_objects_smaller_than:
            if (bb_area < args.ignore_objects_smaller_than):
                continue
        if args.ignore_objects_larger_than:
            if (bb_area > args.ignore_objects_larger_than):
                continue

        if not object['label'] in prediction_per_label.keys():
            prediction_per_label[object['label']] = []

        prediction_per_label[object['label']].append({
            'label': object['label'],
            'x': xmin,
            'y': ymin,
            'width': xmax - xmin,
            'height': ymax - ymin,
            'score': score,
        })

    for label in prediction_per_label.keys():
        # With NMS
        if (args.nms):
            # (x_min, y_min, x_max, y_max
            in_boxes = [
                [box['x'], box['y'], box['x'] + box['width'], box['y'] + box['height'], box['score'], 0] for box in prediction_per_label[label]
            ]
            # print('in_boxes', in_boxes)
            out_boxes = non_max_suppression(np.array(in_boxes), iou_threshold=args.nms_iou_threshold)
            # print('out_boxes', out_boxes)
            for i in range(0, len(out_boxes)):
                if out_boxes[i] == False:
                    continue

                bb = prediction_per_label[label][i]

                bbs.append({
                    'label': label,
                    'x': bb['x'],
                    'y': bb['y'],
                    'width': bb['width'],
                    'height': bb['height'],
                })
        else:
            # non-NMS
            for bb in prediction_per_label[label]:
                bbs.append({
                    'label': label,
                    'x': bb['x'],
                    'y': bb['y'],
                    'width': bb['width'],
                    'height': bb['height'],
                })

    if args.propose_actions:
        raw_data_api.set_sample_proposed_changes(project_id=EI_PROJECT_ID, sample_id=data_id, set_sample_proposed_changes_request={
            'jobId': args.propose_actions,
            'proposedChanges': {
                'boundingBoxes': bbs,
                'metadata': new_metadata,
            },
        })
    else:
        raw_data_api.set_sample_bounding_boxes(project_id=EI_PROJECT_ID, sample_id=data_id, sample_bounding_boxes_request={
            'boundingBoxes': bbs
        })
        raw_data_api.set_sample_metadata(project_id=EI_PROJECT_ID, sample_id=data_id, set_sample_metadata_request={
            'metadata': new_metadata
        })

    print(' OK (took ' + str(prediction_time) + 'ms.)')
    print('    Found objects:')
    for p in predictions:
        box = p["box"]
        label = p["label"]
        score = p["score"]
        xmin, ymin, xmax, ymax = box.values()

        object = objects_by_search_for[prediction['label']]

        below_min_threshold_msg = '(below min. threshold)' if score < object['min_confidence'] else ''

        print('        %s (%f), x=%d y=%d w=%d h=%d %s' %
            (label, score, xmin, ymin, (xmax - xmin), (ymax - ymin), below_min_threshold_msg))

print('')
print('All done!')

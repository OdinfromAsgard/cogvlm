import argparse
import torch
import os
import json
import cv2
import time
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--quant", choices=[4], type=int, default=None, help='quantization bits')
parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--bf16", action="store_true")
# parser.add_argument("--gpu", type=int, default=7, help='GPU device index')
args = parser.parse_args()

MODEL_PATH = args.from_pretrained
TOKENIZER_PATH = args.local_tokenizer
DEVICE = f'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)

if args.bf16:
    torch_type = torch.bfloat16
else:
    torch_type = torch.float16

print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

if args.quant:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        trust_remote_code=True
    ).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        bnb_4bit_compute_dtype=torch.float16,
        load_in_4bit=args.quant is not None,
        trust_remote_code=True
    ).to(DEVICE).eval()

def extract_frames(video_path):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps

    # Calculate number of frames to extract; ensure at least 1 frame for very short videos
    num_frames = max(1, int(duration / 2))  # Ensures at least one frame is extracted

    # If only one frame is needed, take the middle frame
    if num_frames == 1:
        frame_indices = [total_frames // 2]
    else:
        frame_indices = [i * (total_frames // num_frames) for i in range(num_frames)]

    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames
##for etri made some changes 
def process_videos(folder_path):
    results = {}
    start_time = time.time()

    # Define the two queries
    queries = [
        "Give a detailed desciption of the actions happening and descibe the scene, include motions and the objects interacted by the person.",
        "Summarize the content of the scene in details explaining all events happening"
    ]

    # Loop through the folders from P001 to P100
    for i in range(1, 101):  # Adjust range to cover P001 to P100
        folder_name = f"P{i:03d}"  # Changed to 'P' prefix and format to three digits
        current_folder_path = os.path.join(folder_path, folder_name)
        if os.path.exists(current_folder_path):
            print(f"Processing folder: {folder_name}")

            for root, dirs, files in os.walk(current_folder_path):
                for file in files:
                    if file.endswith(".mp4") or file.endswith(".avi"):  # Include other video extensions if needed
                        video_path = os.path.join(root, file)
                        print(f"Processing video: {video_path}")
                        frames = extract_frames(video_path)

                        captions = []
                        for j, frame in enumerate(frames):
                            query = queries[j % 2]  # Alternate between the two queries
                            input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[frame])

                            inputs = {
                                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                                'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]]
                            }
                            if 'cross_images' in input_by_model and input_by_model['cross_images']:
                                inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

                            gen_kwargs = {"max_length": 2048, "do_sample": False}
                            with torch.no_grad():
                                outputs = model.generate(**inputs, **gen_kwargs)
                                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                                response = tokenizer.decode(outputs[0])
                                response = response.split("</s>")[0]
                                captions.append(response)

                        results[video_path] = captions
                        # Create the output file path with the folder name
                        output_file = os.path.join(current_folder_path, f"output_json_{folder_name}.json")

                        with open(output_file, "w") as f:
                            json.dump(results, f, indent=4)
                        print(f"Captions saved to {output_file}")
                        elapsed_time = time.time() - start_time
                        print(f"Elapsed time: {elapsed_time:.2f} seconds")
        else:
            print(f"Folder {folder_name} does not exist. Skipping.")

    return results

folder_path = "/data/ETRI_COMBINATION_VIDEOS"
results = process_videos(folder_path)


# import argparse
# import torch
# import os
# import json
# import cv2
# import time
# from PIL import Image
# from transformers import AutoModelForCausalLM, LlamaTokenizer

# parser = argparse.ArgumentParser()
# parser.add_argument("--quant", choices=[4], type=int, default=4, help='quantization bits')
# parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
# parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
# parser.add_argument("--fp16", action="store_true")
# parser.add_argument("--bf16", action="store_true")
# parser.add_argument("--text_file", type=str, required=False, default="/data/CHARADES/Charades/test_submission_caption.txt", help='Path to the text file with video filenames')
# # parser.add_argument("--gpu", type=int, default=7, help='GPU device index')
# args = parser.parse_args()

# MODEL_PATH = args.from_pretrained
# TOKENIZER_PATH = args.local_tokenizer
# DEVICE = f'cuda' if torch.cuda.is_available() else 'cpu'

# tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)

# if args.bf16:
#     torch_type = torch.bfloat16
# else:
#     torch_type = torch.float16

# print("========Use torch type as:{} with device:{}========\n\n".format(torch_type, DEVICE))

# if args.quant:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch_type,
#         low_cpu_mem_usage=True,
#         load_in_4bit=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         trust_remote_code=True
#     ).eval()
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH,
#         torch_dtype=torch_type,
#         low_cpu_mem_usage=True,
#         bnb_4bit_compute_dtype=torch.float16,
#         load_in_4bit=args.quant is not None,
#         trust_remote_code=True
#     ).to(DEVICE).eval()

# def extract_frames(video_path, max_frames=16):
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     duration = total_frames / fps
#     num_frames = min(int(duration / 2), max_frames)  # Extract half the number of frames based on duration but no more than max_frames

#     frame_indices = [i * (total_frames // num_frames) for i in range(num_frames)]
#     for i in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(Image.fromarray(frame))
#     cap.release()
#     return frames

# def process_videos(folder_path,text_file_path):
#     results = {}
#     start_time = time.time()
#     # Read filenames from the text file
#     with open(text_file_path, 'r') as f:
#         video_filenames = [line.split()[0] + '.mp4' for line in f.readlines()]


#     # Define the two queries
#     queries = [
#         "Give a detailed desciption of the actions happening and descibe the image, include motions and the objects interacted by the person.",
#         "Summarize the content of the image in details explaining all events happening."
#     ]

#     if os.path.exists(folder_path):
#         print(f"Processing folder: {folder_path}")

#         for root, dirs, files in os.walk(folder_path):
#             for file in files:
#                 if file in video_filenames:
#                     if file.endswith(".mp4") or file.endswith(".avi"):  # Add more video file extensions if needed
#                         video_path = os.path.join(root, file)
#                         print(f"Processing video: {video_path}")
#                         frames = extract_frames(video_path)

#                         captions = []
#                         for j, frame in enumerate(frames):
#                             query = queries[j % 2]  # Alternate between the two queries
#                             input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[frame])

#                             inputs = {
#                                 'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
#                                 'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
#                                 'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
#                                 'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]]
#                             }
#                             if 'cross_images' in input_by_model and input_by_model['cross_images']:
#                                 inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

#                             gen_kwargs = {"max_length": 2048, "do_sample": False}
#                             with torch.no_grad():
#                                 outputs = model.generate(**inputs, **gen_kwargs)
#                                 outputs = outputs[:, inputs['input_ids'].shape[1]:]
#                                 response = tokenizer.decode(outputs[0])
#                                 response = response.split("</s>")[0]
#                                 captions.append(response)

#                         results[video_path] = captions
#                         # Create the output file path with the folder name
#                         output_file = os.path.join(folder_path, f"cogvlm_generated_captions.json")

#                         with open(output_file, "w") as f:
#                             json.dump(results, f, indent=4)
#                         print(f"Captions saved to {output_file}")
#                         elapsed_time = time.time() - start_time
#                         print(f"Elapsed time: {elapsed_time:.2f} seconds")
#         else:
#             print(f"Folder {folder_path} does not exist. Skipping.")

#     return results
# text_file_path = args.text_file 
# folder_path = input("Enter the folder path containing videos: ")
# results = process_videos(folder_path,text_file_path)



# import argparse
# import torch
# import os
# import json
# import cv2
# import time
# from PIL import Image
# from transformers import AutoModelForCausalLM, LlamaTokenizer

# # Initialize parser and add arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--quant", choices=[4], type=int, default=4, help='quantization bits')
# parser.add_argument("--from_pretrained", type=str, default="THUDM/cogagent-chat-hf", help='pretrained ckpt')
# parser.add_argument("--local_tokenizer", type=str, default="lmsys/vicuna-7b-v1.5", help='tokenizer path')
# parser.add_argument("--fp16", action="store_true")
# parser.add_argument("--bf16", action="store_true")
# parser.add_argument("--text_file", type=str, required=False,default="/data/CHARADES/Charades/test_submission_caption.txt", help='Path to the text file with video filenames')
# args = parser.parse_args()

# # Set model and tokenizer paths, device configuration
# MODEL_PATH = args.from_pretrained
# TOKENIZER_PATH = args.local_tokenizer
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# tokenizer = LlamaTokenizer.from_pretrained(TOKENIZER_PATH)

# # Determine data type for model
# torch_type = torch.bfloat16 if args.bf16 else torch.float16
# print(f"========Use torch type as:{torch_type} with device:{DEVICE}========\n\n")

# # Load model with or without quantization
# if args.quant:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH, torch_dtype=torch_type, low_cpu_mem_usage=True,
#         load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, trust_remote_code=True
#     ).eval()
# else:
#     model = AutoModelForCausalLM.from_pretrained(
#         MODEL_PATH, torch_dtype=torch_type, low_cpu_mem_usage=True,
#         bnb_4bit_compute_dtype=torch.float16, load_in_4bit=args.quant is not None,
#         trust_remote_code=True
#     ).to(DEVICE).eval()

# # Function to extract frames limiting to a maximum of 16 frames
# def extract_frames(video_path, max_frames=16):
#     frames = []
#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     duration = total_frames / fps
#     num_frames = min(int(duration / 2), max_frames)  # Extract half the number of frames based on duration but no more than max_frames

#     frame_indices = [i * (total_frames // num_frames) for i in range(num_frames)]
#     for i in frame_indices:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, i)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frames.append(Image.fromarray(frame))
#     cap.release()
#     return frames

# # Function to process videos and generate captions
# def process_videos(folder_path, text_file_path):
#     results = {}
#     start_time = time.time()
#     output_file = os.path.join(folder_path, "output_json.json")
#     existing_results = {}

#     # Load existing results if the output file already exists
#     if os.path.exists(output_file):
#         with open(output_file, "r") as f:
#             existing_results = json.load(f)

#     with open(text_file_path, 'r') as f:
#         video_filenames = [line.strip() for line in f.readlines()]

#     queries = [
#         "Give a detailed description of the actions happening and describe the image, include motions and the objects interacted by the person.",
#         "Summarize the content of the image in details explaining all events happening."
#     ]

#     if os.path.exists(folder_path):
#         print(f"Processing folder: {folder_path}")
#         for root, dirs, files in os.walk(folder_path):
#             for file in files:
#                 if file in video_filenames:
#                     video_path = os.path.join(root, file)
#                     # Check if the video has already been processed and captions are available
#                     if video_path in existing_results:
#                         print(f"Skipping {video_path}, already processed.")
#                         continue

#                     print(f"Processing video: {video_path}")
#                     frames = extract_frames(video_path)

#                     captions = []
#                     for j, frame in enumerate(frames):
#                         query = queries[j % 2]
#                         input_by_model = model.build_conversation_input_ids(tokenizer, query=query, history=[], images=[frame])

#                         inputs = {
#                             'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
#                             'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
#                             'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
#                             'images': [[input_by_model['images'][0].to(DEVICE).to(torch_type)]]
#                         }
#                         if 'cross_images' in input_by_model and input_by_model['cross_images']:
#                             inputs['cross_images'] = [[input_by_model['cross_images'][0].to(DEVICE).to(torch_type)]]

#                         gen_kwargs = {"max_length": 2048, "do_sample": False}
#                         with torch.no_grad():
#                             outputs = model.generate(**inputs, **gen_kwargs)
#                             outputs = outputs[:, inputs['input_ids'].shape[1]:]
#                             response = tokenizer.decode(outputs[0])
#                             response = response.split("</s>")[0]
#                             captions.append(response)

#                     results[video_path] = captions

#         # Save combined new and existing results
#         with open(output_file, "w") as f:
#             json.dump({**existing_results, **results}, f, indent=4)
#         print(f"Captions saved to {output_file}")
#         elapsed_time = time.time() - start_time
#         print(f"Elapsed time: {elapsed_time:.2f} seconds")
#     else:
#         print(f"Folder {folder_path} does not exist. Skipping.")

#     return results


# text_file_path = args.text_file
# folder_path = input("Enter the folder path containing videos: ")
# results = process_videos(folder_path, text_file_path)

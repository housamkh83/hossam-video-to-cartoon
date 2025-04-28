# video_to_cartoon_gui.py (PyTorch Version - Requires CORRECT network.py)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import subprocess
import cv2
import numpy as np
import threading
import queue
import time
import shutil
from PIL import Image # للتعامل مع الصور بشكل أسهل مع Torchvision
import sys # لاستخدامه في تحديد مسار السكربت

# --- إعدادات PyTorch ---
# !!! هام: يجب تحميل ملف الموديل .pth ووضعه بجانب السكربت
# --- !!! هذا السطر يحدد المسار نسبة لمكان السكربت !!! ---
script_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(script_dir, 'Hayao_net_G_float.pth') # <--- تأكد من اسم الملف

# !!! هام جداً: يجب تحميل ملف تعريف الشبكة الصحيح (network.py) ووضعه بجانب السكربت
# !!! يجب تثبيت: PyTorch و Torchvision (انظر https://pytorch.org/)
try:
    import torch
    import torchvision.transforms as transforms
    # --- !!! هذا السطر الحاسم يتطلب وجود ملف network.py الصحيح بنفس المجلد !!! ---
    from network import Generator  # <--- استيراد تعريف بنية الموديل الصحيح
    PYTORCH_AVAILABLE = True
    MODEL_DEF_AVAILABLE = True
except ImportError as e:
    PYTORCH_AVAILABLE = False
    # تحقق إذا كان الخطأ بسبب عدم وجود network.py أو مشكلة في torch/torchvision
    if 'network' in str(e) or 'Generator' in str(e):
         MODEL_DEF_AVAILABLE = False
         print("خطأ حاسم: لم يتم العثور على ملف تعريف الموديل (network.py) أو الكلاس Generator بداخله. يرجى تحميل الملف الصحيح ووضعه بجانب السكربت.")
    else:
        MODEL_DEF_AVAILABLE = True # نفترض أن تعريف الموديل قد يكون متاحًا لكن المشكلة في بايتورش
        print(f"تحذير: خطأ في استيراد PyTorch أو مكوناته: {e}. تأكد من تثبيت torch و torchvision.")
        print("لن يعمل تحويل CartoonGAN.")
except Exception as e: # Catch other potential errors during import
    PYTORCH_AVAILABLE = False
    MODEL_DEF_AVAILABLE = False
    print(f"خطأ غير متوقع أثناء استيراد المكتبات: {e}")


# --- متغيرات عامة ---
video_path = ""
conversion_thread = None
stop_event = threading.Event()

# --- وظائف مساعدة ---
# def resource_path(relative_path): # قد لا نحتاجها إذا استخدمنا __file__
#     try:
#         base_path = sys._MEIPASS
#     except Exception:
#         base_path = os.path.abspath(".")
#     return os.path.join(base_path, relative_path)

def create_output_dirs(base_folder="output"):
    """ إنشاء مجلدات الإخراج داخل مجلد رئيسي بجانب السكربت """
    script_dir = os.path.dirname(os.path.abspath(__file__)) # المجلد الحالي للسكربت
    main_output_dir = os.path.join(script_dir, base_folder) # مجلد output الرئيسي
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(main_output_dir, f"conversion_{timestamp}") # مجلد لهذه العملية
    frames_dir = os.path.join(output_dir, "frames")
    cartoon_frames_dir = os.path.join(output_dir, "cartoon_frames")
    os.makedirs(output_dir, exist_ok=True) # سيقوم بانشاء main_output_dir إذا لم يكن موجوداً
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(cartoon_frames_dir, exist_ok=True)
    return output_dir, frames_dir, cartoon_frames_dir

def select_video():
    global video_path
    path = filedialog.askopenfilename(
        title="اختر ملف فيديو",
        filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")]
    )
    if path:
        video_path = path
        label_status.config(text=f"تم اختيار: {os.path.basename(video_path)}")
        # تفعيل زر التحويل فقط إذا كانت المكتبات والموديل جاهزة
        if PYTORCH_AVAILABLE and MODEL_DEF_AVAILABLE and os.path.exists(MODEL_PATH):
             btn_convert.config(state=tk.NORMAL)
        else:
             btn_convert.config(state=tk.DISABLED)
             check_pytorch_and_model(show_error=False) # عرض تحذير في الحالة
    else:
        label_status.config(text="لم يتم اختيار فيديو.")
        btn_convert.config(state=tk.DISABLED)

def check_ffmpeg():
    if shutil.which("ffmpeg") is None:
        messagebox.showerror("خطأ FFMpeg", "لم يتم العثور على برنامج ffmpeg.\nيرجى تثبيته وإضافته إلى متغيرات البيئة (PATH) أو وضعه بجانب البرنامج.")
        return False
    return True

def check_pytorch_and_model(show_error=True):
    # يتم التحقق من PYTORCH_AVAILABLE و MODEL_DEF_AVAILABLE بناءً على نتيجة الاستيراد الأولي
    if not PYTORCH_AVAILABLE:
        # رسالة الخطأ المحددة تم طباعتها بالفعل أثناء الاستيراد الأولي
        if show_error: messagebox.showerror("خطأ PyTorch", "مكتبة PyTorch أو Torchvision غير مثبتة أو حدث خطأ عند استيرادها.\nراجع الرسائل في الطرفية (console).")
        else: label_status.config(text="تحذير: PyTorch/Torchvision غير متاحة.")
        return False
    if not MODEL_DEF_AVAILABLE:
         # رسالة الخطأ المحددة تم طباعتها بالفعل أثناء الاستيراد الأولي
         if show_error: messagebox.showerror("خطأ تعريف الموديل", "ملف تعريف الموديل (network.py) غير موجود أو لا يمكن استيراد الكلاس Generator منه.\nراجع الرسائل في الطرفية (console).")
         else: label_status.config(text="تحذير: ملف تعريف الموديل (network.py) مفقود أو غير صحيح.")
         return False
    # التحقق من وجود ملف الأوزان .pth
    if not os.path.exists(MODEL_PATH):
        if show_error: messagebox.showerror("خطأ ملف الموديل", f"ملف الموديل (.pth) غير موجود!\nالمسار المتوقع: {MODEL_PATH}\n\nقم بتحميل ملف .pth وضعه في نفس مجلد السكربت.")
        else: label_status.config(text=f"تحذير: ملف الموديل ({os.path.basename(MODEL_PATH)}) مفقود.")
        return False
    return True

# --- وظيفة التحويل (تعمل في Thread منفصل) ---
def run_conversion(q, vid_path):
    global stop_event
    stop_event.clear()

    # 1. التحقق المبدئي (تم نقله إلى start_conversion ليعرض الخطأ مباشرة)
    q.put(("status", "جاري التحضير..."))
    q.put(("progress", 0))

    # 2. إنشاء مجلدات الإخراج
    try:
        output_dir, frames_dir, cartoon_frames_dir = create_output_dirs()
    except Exception as e:
        q.put(("status", f"خطأ في إنشاء المجلدات: {e}"))
        q.put(("finished", False))
        return

    # 3. حساب FPS واستخراج الإطارات
    try:
        q.put(("status", "جاري حساب FPS..."))
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened(): raise Exception("لا يمكن فتح ملف الفيديو.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count_approx = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fps <= 0: fps = 25
        q.put(("status", f"FPS: {fps:.2f} - الأبعاد: {width}x{height}"))

        q.put(("status", "جاري استخراج الإطارات..."))
        frame_filename_pattern = os.path.join(frames_dir, "frame_%06d.png")
        # التأكد من أن ffmpeg متاح قبل محاولة استخدامه
        if not check_ffmpeg():
            raise Exception("FFmpeg غير موجود أو غير متاح في الـ PATH.")

        ffmpeg_command = [
            'ffmpeg', '-i', vid_path, '-vf', f'fps={fps}',
            '-vsync', 'vfr', frame_filename_pattern
        ]
        # إخفاء نافذة ffmpeg على ويندوز
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, startupinfo=startupinfo)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
             print("FFmpeg stderr:", stderr)
             # محاولة تقديم رسالة خطأ أوضح قليلاً
             error_message = f"حدث خطأ أثناء استخراج الإطارات بواسطة ffmpeg.\nالرمز: {process.returncode}"
             if "No such file or directory" in stderr:
                 error_message += "\nتأكد من أن مسار الفيديو صحيح ولا يحتوي على حروف خاصة جداً."
             elif "Permission denied" in stderr:
                 error_message += "\nتحقق من أذونات القراءة للملف والكتابة للمجلد."
             else:
                  error_message += f"\n{stderr[:500]}..." # عرض جزء من الخطأ
             raise Exception(error_message)


        extracted_frames = sorted(os.listdir(frames_dir))
        num_frames = len(extracted_frames)
        if num_frames == 0: raise Exception("فشل استخراج الإطارات أو لا توجد إطارات في الفيديو.")
        q.put(("status", f"تم استخراج {num_frames} إطار."))

    except Exception as e:
        q.put(("status", f"خطأ أثناء استخراج الإطارات: {e}"))
        q.put(("finished", False))
        return

    # 4. تحميل موديل PyTorch
    model = None
    try:
        q.put(("status", "جاري تحميل موديل PyTorch..."))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q.put(("status", f"استخدام الجهاز: {device}"))

        # --- !!! هنا يتم استخدام الكلاس Generator المستورد من network.py الصحيح !!! ---
        model = Generator()
        # تحميل الأوزان
        # استخدام strict=False قد يتجاوز أخطاء المفاتيح المفقودة/الزائدة لكنه ليس حلاً للمشكلة الأساسية!
        # الأفضل هو التأكد من تطابق network.py مع ملف .pth
        try:
             # محاولة التحميل الصارمة أولاً (strict=True افتراضياً)
             model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        except RuntimeError as load_error:
             # إذا فشل التحميل الصارم بسبب عدم تطابق المفاتيح
             if 'Missing key(s)' in str(load_error) or 'Unexpected key(s)' in str(load_error):
                  error_msg = f"خطأ في تحميل الموديل:\nعدم تطابق بين بنية الموديل في 'network.py' والأوزان في '{os.path.basename(MODEL_PATH)}'.\n"
                  error_msg += "يرجى التأكد من استخدام ملف 'network.py' الصحيح المطابق لهذا الموديل.\n\n"
                  error_msg += f"تفاصيل الخطأ:\n{load_error}"
                  q.put(("status", error_msg)) # عرض رسالة الخطأ الكاملة للمستخدم
                  q.put(("finished", False))
                  return # إيقاف التنفيذ لأن الموديل لم يتم تحميله بشكل صحيح
             else:
                  raise load_error # إذا كان الخطأ من نوع آخر، أعد رفعه

        model.to(device)
        model.eval() # ضروري جداً
        q.put(("status", "تم تحميل موديل PyTorch بنجاح."))

        # تعريف خطوات التحويل (Preprocessing) - تأكد من أنها مناسبة للموديل
        img_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    except Exception as e:
        q.put(("status", f"خطأ في تحميل أو إعداد موديل PyTorch: {e}"))
        q.put(("finished", False))
        return

    # 5. معالجة الإطارات (PyTorch Inference)
    q.put(("status", "جاري تحويل الإطارات إلى كرتون (قد تكون العملية بطيئة)..."))
    start_time = time.time()
    processed_count = 0
    try:
        with torch.no_grad(): # مهم جداً
            for i, frame_name in enumerate(extracted_frames):
                if stop_event.is_set():
                    q.put(("status", "تم الإيقاف بواسطة المستخدم."))
                    q.put(("finished", False))
                    return

                img_path = os.path.join(frames_dir, frame_name)
                try:
                    img_pil = Image.open(img_path).convert('RGB')
                except Exception as img_err:
                     print(f"تحذير: فشل قراءة الإطار {frame_name} باستخدام PIL: {img_err}, سيتم تخطيه.")
                     continue # انتقل للإطار التالي

                # تطبيق التحويلات
                input_tensor = img_transforms(img_pil).unsqueeze(0).to(device)

                # Inference
                output_tensor = model(input_tensor)

                # Postprocessing
                output_image_np = output_tensor.squeeze(0).cpu().numpy()
                output_image_np = output_image_np * 0.5 + 0.5
                output_image_np = np.transpose(output_image_np, (1, 2, 0))
                output_image_np = (output_image_np * 255.0)
                output_image_np = np.clip(output_image_np, 0, 255).astype(np.uint8)
                cartoon_img_bgr = cv2.cvtColor(output_image_np, cv2.COLOR_RGB2BGR)

                # حفظ الإطار الكرتوني
                cartoon_frame_path = os.path.join(cartoon_frames_dir, frame_name)
                cv2.imwrite(cartoon_frame_path, cartoon_img_bgr)
                processed_count += 1

                # تحديث التقدم
                progress_percent = (i + 1) * 100 / num_frames
                elapsed_time = time.time() - start_time
                est_total_time = (elapsed_time / (i + 1)) * num_frames if (i + 1) > 0 else 0
                est_remaining = est_total_time - elapsed_time
                q.put(("progress", progress_percent))
                q.put(("status", f"معالجة الإطار {i+1}/{num_frames} - متبقي: {time.strftime('%H:%M:%S', time.gmtime(est_remaining))}"))

    except Exception as e:
        q.put(("status", f"خطأ أثناء معالجة الإطارات بـ PyTorch: {e}"))
        q.put(("finished", False))
        return
    finally:
        # محاولة تحرير الذاكرة
        if model is not None: del model
        if 'input_tensor' in locals(): del input_tensor
        if 'output_tensor' in locals(): del output_tensor
        if torch.cuda.is_available(): torch.cuda.empty_cache()


    # 6. إعادة تجميع الفيديو
    q.put(("status", "جاري إعادة تجميع الفيديو النهائي..."))
    q.put(("progress", 100))
    try:
        video_basename = os.path.splitext(os.path.basename(vid_path))[0]
        output_video_path = os.path.join(output_dir, f"{video_basename}_cartoon_pytorch.mp4")

        # الحصول على أبعاد أول إطار كرتوني
        first_cartoon_frame_path = os.path.join(cartoon_frames_dir, extracted_frames[0])
        first_frame = cv2.imread(first_cartoon_frame_path)
        if first_frame is None: raise Exception("لا يمكن قراءة الإطار الكرتوني الأول.")
        out_height, out_width, _ = first_frame.shape

        # التأكد من وجود ffmpeg مرة أخرى قبل التجميع
        if not check_ffmpeg():
            raise Exception("FFmpeg غير موجود أو غير متاح في الـ PATH.")

        reassemble_command = [
            'ffmpeg', '-framerate', str(fps),
            '-i', os.path.join(cartoon_frames_dir, "frame_%06d.png"),
            '-i', vid_path, '-map', '0:v:0', '-map', '1:a:0?',
            '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p',
            '-c:a', 'aac', '-b:a', '128k', '-shortest', '-y', output_video_path
        ]

        q.put(("status", "تشغيل FFMpeg لتجميع الفيديو..."))
        # إخفاء نافذة ffmpeg
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        process = subprocess.Popen(reassemble_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, startupinfo=startupinfo)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print("FFmpeg Reassemble stderr:", stderr)
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0 :
                 q.put(("status", f"تحذير: خطأ بنسخ الصوت؟ الفيديو هنا: {output_video_path}"))
                 q.put(("finished", True, output_dir))
            else:
                 raise Exception(f"فشل FFMpeg في تجميع الفيديو.\n{stderr[:500]}...")
        else:
             q.put(("status", f"✅ اكتمل التحويل! الفيديو والمجلدات هنا: {output_dir}"))
             q.put(("finished", True, output_dir))

    except Exception as e:
        q.put(("status", f"خطأ أثناء إعادة تجميع الفيديو: {e}"))
        q.put(("finished", False))
        return

    # 7. (اختياري) حذف المجلدات المؤقتة
    # try:
    #     q.put(("status", "جاري حذف الإطارات المؤقتة..."))
    #     shutil.rmtree(frames_dir)
    #     # shutil.rmtree(cartoon_frames_dir) # قد ترغب في إبقاء الإطارات الكرتونية
    # except Exception as e:
    #     q.put(("status", f"تحذير: لم يتم حذف المجلدات المؤقتة: {e}"))

def start_conversion():
    global conversion_thread, stop_event

    # --- التحقق الأولي قبل البدء ---
    if not check_ffmpeg(): return
    if not check_pytorch_and_model(show_error=True): return # يتحقق من كل شيء (torch, network.py, model.pth)

    if conversion_thread and conversion_thread.is_alive():
        messagebox.showwarning("قيد التشغيل", "عملية تحويل أخرى قيد التشغيل بالفعل.")
        return

    if not video_path:
        messagebox.showerror("خطأ", "يرجى اختيار ملف فيديو أولاً.")
        return

    # تعطيل وتفعيل الأزرار
    btn_select.config(state=tk.DISABLED)
    btn_convert.config(state=tk.DISABLED)
    btn_stop.config(state=tk.NORMAL)
    progress_bar['value'] = 0
    label_status.config(text="بدء عملية التحويل (PyTorch)...")

    # بدء الـ Thread
    conversion_thread = threading.Thread(target=run_conversion, args=(update_queue, video_path), daemon=True)
    conversion_thread.start()
    root.after(100, check_queue) # بدء مراقبة الـ Queue

def stop_conversion():
    global stop_event
    if conversion_thread and conversion_thread.is_alive():
        print("إرسال إشارة الإيقاف...")
        stop_event.set()
        btn_stop.config(state=tk.DISABLED)
        label_status.config(text="جاري الإيقاف...")
    else:
        print("لا توجد عملية لإيقافها.")

def check_queue():
    """ تفحص الـ Queue لتحديث الواجهة من الـ Thread الرئيسي """
    try:
        while True:
            message = update_queue.get_nowait() # الحصول على الرسالة بدون انتظار
            msg_type = message[0]
            msg_data = message[1]

            if msg_type == "status":
                label_status.config(text=msg_data)
                # إذا كانت الرسالة تحتوي على تفاصيل خطأ تحميل الموديل، اعرضها في messagebox
                if "خطأ في تحميل الموديل" in msg_data and len(msg_data) > 100: # تحقق من أنها رسالة الخطأ الطويلة
                    messagebox.showerror("خطأ تحميل الموديل", msg_data)

            elif msg_type == "progress":
                progress_bar['value'] = msg_data
            elif msg_type == "finished":
                success = msg_data
                output_location = message[2] if len(message) > 2 else None
                if success:
                    messagebox.showinfo("اكتمل", f"تمت عملية التحويل بنجاح!\nالنتائج في المجلد:\n{output_location}")
                # else: # عرض الخطأ الأخير إذا لم يكن قد تم عرضه بالفعل
                #    final_status = label_status.cget("text")
                #    if "خطأ" in final_status and "تحميل الموديل" not in final_status:
                #       messagebox.showerror("خطأ", final_status)


                # إعادة ضبط الواجهة
                btn_select.config(state=tk.NORMAL)
                # أعد تفعيل زر التحويل فقط إذا كان الفيديو لا يزال محدداً والمكتبات جاهزة
                if video_path and PYTORCH_AVAILABLE and MODEL_DEF_AVAILABLE and os.path.exists(MODEL_PATH):
                    btn_convert.config(state=tk.NORMAL)
                else:
                     btn_convert.config(state=tk.DISABLED)
                btn_stop.config(state=tk.DISABLED)
                progress_bar['value'] = 0
                # stop_event.clear() # لا تمسحها هنا، امسحها في بداية run_conversion
                return # أوقف التحقق من الـ Queue لهذه الدورة

    except queue.Empty:
        # إذا كانت الـ Queue فارغة، استمر في التحقق لاحقاً إذا كان الـ Thread لا يزال يعمل
        if conversion_thread and conversion_thread.is_alive():
            root.after(100, check_queue)
        else: # Thread انتهى ولم تكن هناك رسالة "finished" (غير متوقع لكن احتياطي)
              btn_select.config(state=tk.NORMAL)
              if video_path and PYTORCH_AVAILABLE and MODEL_DEF_AVAILABLE and os.path.exists(MODEL_PATH): btn_convert.config(state=tk.NORMAL)
              else: btn_convert.config(state=tk.DISABLED)
              btn_stop.config(state=tk.DISABLED)


# --- واجهة المستخدم (Tkinter) ---
root = tk.Tk()
root.title("CartoonifyVideo - تحويل الفيديو إلى كرتون (PyTorch)")
root.geometry("500x250")

# استخدام Style لتحسين المظهر قليلاً (اختياري)
style = ttk.Style()
# يمكنك تجربة ثيمات مختلفة مثل 'clam', 'alt', 'default', 'vista' على ويندوز
# style.theme_use('vista')

# إطار للمحتوى
frame = ttk.Frame(root, padding="10")
frame.pack(expand=True, fill=tk.BOTH)

# الأزرار والعناصر
btn_select = ttk.Button(frame, text="📂 اختيار فيديو", command=select_video)
btn_select.pack(pady=10, fill=tk.X)

# زر التحويل يبدأ معطلاً
btn_convert = ttk.Button(frame, text="🚀 تحويل إلى كرتون (PyTorch)", command=start_conversion, state=tk.DISABLED)
btn_convert.pack(pady=5, fill=tk.X)

btn_stop = ttk.Button(frame, text="🛑 إيقاف", command=stop_conversion, state=tk.DISABLED)
btn_stop.pack(pady=5, fill=tk.X)

progress_bar = ttk.Progressbar(frame, orient="horizontal", length=300, mode="determinate")
progress_bar.pack(pady=10, fill=tk.X)

label_status = ttk.Label(frame, text="يرجى اختيار ملف فيديو للبدء.", anchor=tk.CENTER, wraplength=480)
label_status.pack(pady=10, fill=tk.X)

# --- بدء التشغيل ---
update_queue = queue.Queue() # إنشاء الـ Queue لتواصل الـ Threads

# التحقق الأولي عند بدء التشغيل وعرض الحالة
check_pytorch_and_model(show_error=False) # لا تعرض messagebox، فقط تحدث label_status

root.mainloop()
import cv2
import os



def save_performance(filepath, message, fps_array, max_seconds, cap0, cap1=None):
	file_name = os.path.basename(filepath)
	filename = file_name.split('.')[0]
	print('Saving...')
	with open(f"output/timestamps/{filename}", "w") as f:
		f.write("Camera 0 parameters\n")
		f.write(f" - Width  {cap0.get(3)}\n")
		f.write(f" - Height {cap0.get(4)}\n")
		f.write(f" - FPS    {cap0.get(5)}\n")
		if cap1!=None:
			f.write("Camera 1 parameters\n")
			f.write(f" - Width  {cap1.get(3)}\n")
			f.write(f" - Height {cap1.get(4)}\n")
			f.write(f" - FPS    {cap1.get(5)}\n")
		
		f.write(f"{message} during {max_seconds} seconds.\n")
		for fps in fps_array:
			f.write(f"{fps}\n")
			total += fps
		avg = total / len(fps_array)
		print(f"Average FPS = {avg}")


	
def show_text(cv2, frame, seconds_passed, max_seconds, fps):
	cv2.putText(frame, f'Second = {seconds_passed}/{max_seconds} | FPS = {fps}', (20,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2, cv2.LINE_AA)



def get_max_seconds():
	return 20



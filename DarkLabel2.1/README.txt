DarkLabel (Ver2.1)
- Video/image object labeling & annotation tool
- Video splitting (into images) & image merging (into a video file) tool
- Video/image privacy masking tool (e.g. degrade boxed human area in the image)
- last modified: Sept. 10, 2020

Main Features
- bounding box labeling of video & image objects
- support predefined data formats (pascal_voc/imagenet, darknet yolo)
- support user-defined data formats (define and add your own data format in darklabel.yml)
- automatic box labeling by visual tracking
- semi-automatic labeling by linear interpolation
- user-defined hotkeys and zoom in / zoom out support
- export annotation result as videos or images (it can be used to split a video file into images or vice versa)
- privacy masking of annotated regions by resolution degradation (export annotation result after enabling 'apply_bbox_mosaic' parameter in darklabel.yml)
- windows only (32/64 bits)

How to Use
    darklabel.yml: edit this file to define data formats and hotkeys
    
	Arrow/PgUp/PgDn/Home/End: navigate image frames
	
	Mouse: Left(create box), Right(cancel latest)
	Shift+Mouse: Left(modify box), Right(delete selected/all)
	Shift+DoubleClick: modify box properties (label, ID, difficulty)
	
	Ctrl+DoubleClick: select/deselect box trajectory
	*box trajectory: linked boxes over frames with the same ID and label
	
	Ctrl+'+'/'-': zoom in/out
	Ctrl+Arrow: scroll zoomed window
	Ctrl+MouseWheel: zoom in/out
	Ctrl+Mouse: scroll zoomed window
	
	Enter: apply tracking (newly created boxes only)
	Ctrl+'s': save gt
	F1: help

website & download: http://darkpgmr.tistory.com/16
contact: darkpgmr.lee@gmail.com

class InferlessPythonModel:
  def initialize(self):
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
      self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

  def process_image(self,masks,raw_image):
      mask = masks[0].squeeze()
      mask = mask[0].cpu().detach()

      max_value = mask.max()
      if max_value > 0:  # Ensure division is safe
          mask_normalized = mask / max_value
      else:
         mask_normalized = torch.zeros(mask.size())

      # Convert mask_normalized to a numpy array if it's not already
      mask_normalized_np = mask_normalized.cpu().detach().numpy()
      color = np.array([30, 144, 255])  # RGB color for the mask
      opacity = 0.6  # Opacity of the mask
      image_array = np.array(raw_image)
      image_array_normalized = image_array / 255.0
      mask_rgb = np.zeros((*mask_normalized_np.shape, 3), dtype=np.float32)
      for i in range(3):  # Apply the color to the mask
          mask_rgb[..., i] = mask_normalized_np * color[i] / 255

      combined_image = (1 - opacity) * image_array_normalized + opacity * mask_rgb
      combined_image = np.clip(combined_image, 0, 1)  # Ensure the combined image is within the correct range
      combined_image_uint8 = (combined_image * 255).astype(np.uint8)
      img_to_save = Image.fromarray(combined_image_uint8)
      img_to_save.save('combined_image.png')
      buff = BytesIO()
      img_to_save.save(buff, format="PNG")
      img_str = base64.b64encode(buff.getvalue())
      base64_string = img_str.decode('utf-8')
      return img_str.decode('utf-8')

  def infer(self,inputs):
      # "https://huggingface.co/ybelkada/segment-anything/resolve/main/assets/car.png"
      # [450, 600]
      input_points = [[inputs["input_points"]]]
      img_url = inputs["image_url"]
      raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
      inputs = self.processor(raw_image, return_tensors="pt").to(self.device)
      image_embeddings = self.model.get_image_embeddings(inputs["pixel_values"])
      inputs = self.processor(raw_image, input_points=input_points, return_tensors="pt").to(self.device)

      inputs.pop("pixel_values", None)
      inputs.update({"image_embeddings": image_embeddings})

      with torch.no_grad():
          outputs = self.model(**inputs)

      masks = self.processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
      image_data = self.process_image(masks,raw_image)
      return {"generated_image_base64": image_data}


  def finalize(self,args):
      pass

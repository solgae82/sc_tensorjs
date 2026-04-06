<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>모델 사용하기</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
</head>
<body>
	<script>
		var 내일예상온도 = 22;
		tf.loadLayersModel('./model/my-model.json').then(function(model){
			
			var 내일판매량예측 = model.predict(tf.tensor([내일예상온도])).arraySync()[0][0]
			console.log(내일판매량예측); // 43.995765686035156
		});
		 
	</script>
</body>
</html>
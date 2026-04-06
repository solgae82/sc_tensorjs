<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>보스톤 모델 사용하기</title>
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
</head>
<body>
	<script>
		/*
		var 보스톤_원인 = [
		    [0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98],
		    [0.02731,0,7.07,0,0.469,6.421,78.9,4.9671,2,242,17.8,396.9,9.14]
		 ]
		 var 보스톤_결과 = [[24],
			    [21.6]];
		*/
		var model2;
		var 보스톤_집1 = [0.00632,18,2.31,0,0.538,6.575,65.2,4.09,1,296,15.3,396.9,4.98];
		tf.loadLayersModel('./model/boston-model.json').then(function(model){
			
			var 중앙_결과집값 = model.predict(tf.tensor([보스톤_집1])).arraySync()[0][0]
			console.log(중앙_결과집값); // 29.421030044555664
			model2 = model;//가중치 확인 콘솔 작업을 위해 설정
		});
		 
		// 24 - 29 = -5 차이가 난다
	</script>
</body>
</html>
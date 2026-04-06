<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>레모네이드1</title>
	<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@2.0.0/dist/tf.min.js"></script>
</head>
<body>
	<script>
		//1. 과거의 데이터를 준비합니다.
		var 온도 = [20,21,22,23];
		var 판매량 = [40,42,44,46];
		
		var 원인 = tf.tensor(온도); //독립변수
		var 결과 = tf.tensor(판매량); //종속변수
		
		//2. 모델의 모양을 만듭니다.
		var X = tf.input({shape: [1]});
		var Y = tf.layers.dense({units: 1}).apply(X);
		var model = tf.model({inputs: X , outputs: Y});
		
		var compileParam ={
				optimizer: tf.train.adam(),
				loss: tf.losses.meanSquaredError};
		
		model.compile(compileParam);
		
		//3. 데이터로 모델을 학습 시킵니다
		//var fitParam = {epochs: 12000};
		var fitParam = {
			epochs: 2000,
			callbacks:{
				onEpochEnd:function(epoch, logs){
					console.log(
							'epoch 횟수=>',epoch, 
							'MSE=>',logs.loss,						
							'RMSE=>', Math.sqrt(logs.loss)		
							);
				}
			}
		};
		
		model.fit(원인, 결과, fitParam).then(function(result){
			
			//4. 모델을 이용합니다 
			// 4.1 기존 데이터를 이용 (평가)
			var 예측한결과 = model.predict(원인);
			예측한결과.print();
			
			//model.save('downloads://my-model');
		});
	</script>
</body>
</html>
<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('id')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->id), array('view', 'id'=>$data->id)); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Username')); ?>:</b>
	<?php echo CHtml::encode($data->Username); ?>
	<br />

	

	<b><?php echo CHtml::encode($data->getAttributeLabel('Analista')); ?>:</b>
	<?php echo CHtml::encode($data->Analista); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Proceso')); ?>:</b>
	<?php echo CHtml::encode($data->Proceso); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Es_administrador')); ?>:</b>
	<?php echo CHtml::encode($data->Es_administrador); ?>
	<br />

<!-- echo CHtml::encode($data->getAttributeLabel('password')); ?>:</b>
	echo CHtml::encode($data->password); ?>
	<br />  
-->
</div>

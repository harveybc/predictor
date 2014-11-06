<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('Tipo_Aceite')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->Tipo_Aceite), array('view', 'id'=>$data->id)); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Proceso')); ?>:</b>
	<?php echo CHtml::encode($data->Proceso); ?>
	<br />


</div>

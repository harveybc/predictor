<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('id')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->id), array('view', 'id'=>$data->id)); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('revisado')); ?>:</b>
	<?php echo CHtml::encode($data->revisado); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('fecha_enviado')); ?>:</b>
	<?php echo CHtml::encode($data->fecha_enviado); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('fecha_revisado')); ?>:</b>
	<?php echo CHtml::encode($data->fecha_revisado); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('ruta')); ?>:</b>
	<?php echo CHtml::encode($data->ruta); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('usuario')); ?>:</b>
	<?php echo CHtml::encode($data->usuario); ?>
	<br />


</div>

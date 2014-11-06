<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('id')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->id), array('view', 'id'=>$data->id)); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('usuario')); ?>:</b>
	<?php echo CHtml::encode($data->usuario); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('fecha')); ?>:</b>
	<?php echo CHtml::encode($data->fecha); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('evaluacionGeneral')); ?>:</b>
	<?php echo CHtml::encode($data->evaluacionGeneral); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('pregunta1')); ?>:</b>
	<?php echo CHtml::encode($data->pregunta1); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('pregunta2')); ?>:</b>
	<?php echo CHtml::encode($data->pregunta2); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('pregunta3')); ?>:</b>
	<?php echo CHtml::encode($data->pregunta3); ?>
	<br />

	<?php /*
	<b><?php echo CHtml::encode($data->getAttributeLabel('pregunta4')); ?>:</b>
	<?php echo CHtml::encode($data->pregunta4); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('pregunta5')); ?>:</b>
	<?php echo CHtml::encode($data->pregunta5); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('pregunta6')); ?>:</b>
	<?php echo CHtml::encode($data->pregunta6); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('pregunta7')); ?>:</b>
	<?php echo CHtml::encode($data->pregunta7); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('pregunta8')); ?>:</b>
	<?php echo CHtml::encode($data->pregunta8); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('pregunta9')); ?>:</b>
	<?php echo CHtml::encode($data->pregunta9); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('pregunta10')); ?>:</b>
	<?php echo CHtml::encode($data->pregunta10); ?>
	<br />

	*/ ?>

</div>

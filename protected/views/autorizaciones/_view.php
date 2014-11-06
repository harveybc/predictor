<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('id')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->id), array('view', 'id'=>$data->id)); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('usuario')); ?>:</b>
	<?php echo CHtml::encode($data->usuario); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('documento')); ?>:</b>
	<?php echo CHtml::encode($data->documento); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('operacion')); ?>:</b>
	<?php echo CHtml::encode($data->operacion); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('autorizado')); ?>:</b>
	<?php echo CHtml::encode($data->autorizado); ?>
	<br />


</div>

<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('id')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->id), array('view', 'id'=>$data->id)); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Fecha')); ?>:</b>
	<?php echo CHtml::encode($data->Fecha); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('TAG')); ?>:</b>
	<?php echo CHtml::encode($data->TAG); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('IP')); ?>:</b>
	<?php echo CHtml::encode($data->IP); ?>
	<br />


</div>

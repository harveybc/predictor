<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('id')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->id), array('view', 'id'=>$data->id)); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('codigoSAP')); ?>:</b>
	<?php echo CHtml::encode($data->codigoSAP); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('descripcion')); ?>:</b>
	<?php echo CHtml::encode($data->descripcion); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('padre')); ?>:</b>
	<?php echo CHtml::encode($data->padre); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('supervisor')); ?>:</b>
	<?php echo CHtml::encode($data->supervisor); ?>
	<br />


</div>

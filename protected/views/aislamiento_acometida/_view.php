<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('TAG')); ?>:</b>
        <?php echo CHtml::link(CHtml::encode($data->TAG), array('view', 'id'=>$data->Toma)); ?>
        
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Fecha')); ?>:</b>
	<?php echo CHtml::encode($data->Fecha); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('A050')); ?>:</b>
	<?php echo CHtml::encode($data->A050); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('A1')); ?>:</b>
	<?php echo CHtml::encode($data->A1); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('B050')); ?>:</b>
	<?php echo CHtml::encode($data->B050); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('B1')); ?>:</b>
	<?php echo CHtml::encode($data->B1); ?>
	<br />

	<?php /*
	<b><?php echo CHtml::encode($data->getAttributeLabel('C050')); ?>:</b>
	<?php echo CHtml::encode($data->C050); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('C1')); ?>:</b>
	<?php echo CHtml::encode($data->C1); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('OT')); ?>:</b>
	<?php echo CHtml::encode($data->OT); ?>
	<br />

	*/ ?>

</div>

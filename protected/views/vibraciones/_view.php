<div class="view">

	
	<?php /*echo CHtml::encode($data->getAttributeLabel('Toma')); ?>:</b>
	<?php echo CHtml::encode($data->Toma); */?>
	

	<b><?php echo CHtml::encode($data->getAttributeLabel('TAG')); ?>:</b>
        <?php echo CHtml::link(CHtml::encode($data->TAG), array('view', 'id'=>$data->id)); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Fecha')); ?>:</b>
	<?php echo CHtml::encode($data->Fecha); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('OT')); ?>:</b>
	<?php echo CHtml::encode($data->OT); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('VibLL')); ?>:</b>
	<?php echo CHtml::encode($data->VibLL); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('VibLA')); ?>:</b>
	<?php echo CHtml::encode($data->VibLA); ?>
	<br />

	<?php /*
	<b><?php echo CHtml::encode($data->getAttributeLabel('Temperatura')); ?>:</b>
	<?php echo CHtml::encode($data->Temperatura); ?>
	<br />

	*/ ?>

</div>

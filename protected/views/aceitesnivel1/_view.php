<div class="view">

        <b><?php echo CHtml::encode($data->getAttributeLabel('TAG')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->TAG), array('view', 'id'=>$data->id)); ?>
	<br />
        
	<b><?php echo CHtml::encode($data->getAttributeLabel('Fecha')); ?>:</b>
	<?php echo CHtml::encode($data->Fecha); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('OT')); ?>:</b>
	<?php echo CHtml::encode($data->OT); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Estado')); ?>:</b>
	<?php echo CHtml::encode($data->Estado); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Medicion')); ?>:</b>
	<?php echo CHtml::encode($data->Medicion); ?>
	<br />

	<?php /*
	<b><?php echo CHtml::encode($data->getAttributeLabel('Tipo')); ?>:</b>
	<?php echo CHtml::encode($data->Tipo); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Analista')); ?>:</b>
	<?php echo CHtml::encode($data->Analista); ?>
	<br />
         * <b><?php echo CHtml::encode($data->getAttributeLabel('Toma')); ?>:</b>
	<?php echo CHtml::encode($data->Toma); ?>
	<br />

	*/ ?>

</div>

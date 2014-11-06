<div class="view">

	
	<b><?php echo CHtml::encode($data->getAttributeLabel('descripcion')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->descripcion), array('view', 'id'=>$data->id)); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('permitirAdiciones')); ?>:</b>
	<?php echo CHtml::encode($data->permitirAdiciones); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('permitirAnotaciones')); ?>:</b>
	<?php echo CHtml::encode($data->permitirAnotaciones); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('autorizarOtros')); ?>:</b>
	<?php echo CHtml::encode($data->autorizarOtros); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('requiereAutorizacion')); ?>:</b>
	<?php echo CHtml::encode($data->requiereAutorizacion); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('secuencia')); ?>:</b>
	<?php echo CHtml::encode($data->secuencia); ?>
	<br />

	<?php /*
	<b><?php echo CHtml::encode($data->getAttributeLabel('ordenSecuencia')); ?>:</b>
	<?php echo CHtml::encode($data->ordenSecuencia); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('eliminado')); ?>:</b>
	<?php echo CHtml::encode($data->eliminado); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('conservacionInicio')); ?>:</b>
	<?php echo CHtml::encode($data->conservacionInicio); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('conservacionFin')); ?>:</b>
	<?php echo CHtml::encode($data->conservacionFin); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('conservacionPermanente')); ?>:</b>
	<?php echo CHtml::encode($data->conservacionPermanente); ?>
	<br />

	

	*/ ?>

</div>

<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('id')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->id), array('view', 'id'=>$data->id)); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('metaDoc')); ?>:</b>
	<?php echo CHtml::encode($data->metaDoc); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('cedula')); ?>:</b>
	<?php echo CHtml::encode($data->cedula); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('usuario')); ?>:</b>
	<?php echo CHtml::encode($data->usuario); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('usuarioRcv')); ?>:</b>
	<?php echo CHtml::encode($data->usuarioRcv); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('fechaPrestamo')); ?>:</b>
	<?php echo CHtml::encode($data->fechaPrestamo); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('fechaDevolucion')); ?>:</b>
	<?php echo CHtml::encode($data->fechaDevolucion); ?>
	<br />

	<?php /*
	<b><?php echo CHtml::encode($data->getAttributeLabel('observaciones')); ?>:</b>
	<?php echo CHtml::encode($data->observaciones); ?>
	<br />

	*/ ?>

</div>

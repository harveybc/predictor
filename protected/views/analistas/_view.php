<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('Analista')); ?>:</b>
        <?php echo CHtml::link(CHtml::encode($data->Analista), array('view', 'id'=>$data->id)); ?>
	
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Proceso')); ?>:</b>
	<?php echo CHtml::encode($data->Proceso); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Pto_trabajo')); ?>:</b>
	<?php echo CHtml::encode($data->Pto_trabajo); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('modulo')); ?>:</b>
	<?php echo CHtml::encode($data->modulo); ?>
	<br />


</div>

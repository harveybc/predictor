<div class="view">

	<b><?php echo CHtml::encode($data->getAttributeLabel('id')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->id), array('view', 'id'=>$data->id)); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Fecha')); ?>:</b>
	<?php echo CHtml::encode($data->Fecha); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Eff_Shift')); ?>:</b>
	<?php echo CHtml::encode($data->Eff_Shift); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Count_Fill_Batch')); ?>:</b>
	<?php echo CHtml::encode($data->Count_Fill_Batch); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Count_Pall_Batch')); ?>:</b>
	<?php echo CHtml::encode($data->Count_Pall_Batch); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Count_Fill_Shift')); ?>:</b>
	<?php echo CHtml::encode($data->Count_Fill_Shift); ?>
	<br />


</div>

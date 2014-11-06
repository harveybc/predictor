<div class="view">
        <?php 
            $cod="";
            if (isset($data->Codigo)) $cod=$data->Codigo;
            
        ?>
	<b><?php echo CHtml::encode($data->getAttributeLabel('Codigo')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($cod), array('view', 'id'=>$data->id)); ?>
	<br />
        
	<b><?php echo CHtml::encode($data->getAttributeLabel('TAG')); ?>:</b>
	<?php echo CHtml::link(CHtml::encode($data->TAG), array('view', 'id'=>$data->id)); ?>
	<br />

         <b><?php echo CHtml::encode($data->getAttributeLabel('Proceso')); ?>:</b>
	<?php echo CHtml::encode($data->Proceso); ?>
	<br />
	<b><?php echo CHtml::encode($data->getAttributeLabel('Area')); ?>:</b>
	<?php echo CHtml::encode($data->Area); ?>
	<br />
	<b><?php echo CHtml::encode($data->getAttributeLabel('Equipo')); ?>:</b>
	<?php echo CHtml::encode($data->Equipo); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Motor')); ?>:</b>
	<?php echo CHtml::encode($data->Motor); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('kW')); ?>:</b>
	<?php echo CHtml::encode($data->kW); ?>
	<br />

	<?php /*
	<b><?php echo CHtml::encode($data->getAttributeLabel('Velocidad')); ?>:</b>
	<?php echo CHtml::encode($data->Velocidad); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Marca')); ?>:</b>
	<?php echo CHtml::encode($data->Marca); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Modelo')); ?>:</b>
	<?php echo CHtml::encode($data->Modelo); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Serie')); ?>:</b>
	<?php echo CHtml::encode($data->Serie); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Rod_LC')); ?>:</b>
	<?php echo CHtml::encode($data->Rod_LC); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Rod_LA')); ?>:</b>
	<?php echo CHtml::encode($data->Rod_LA); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Lubricante')); ?>:</b>
	<?php echo CHtml::encode($data->Lubricante); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('IP')); ?>:</b>
	<?php echo CHtml::encode($data->IP); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('Frame')); ?>:</b>
	<?php echo CHtml::encode($data->Frame); ?>
	<br />

	<b><?php echo CHtml::encode($data->getAttributeLabel('PathFoto')); ?>:</b>
	<?php echo CHtml::encode($data->PathFoto); ?>
	<br />

	*/ ?>

</div>

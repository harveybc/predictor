<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

	<div class="row">
		<?php echo $form->labelEx($model,'codigoSAP'); ?>
<?php echo $form->textField($model,'codigoSAP',array('size'=>60,'maxlength'=>64)); ?>
<?php echo $form->error($model,'codigoSAP'); ?>
	</div>

	<div class="row">
		<?php echo $form->labelEx($model,'descripcion'); ?>
<?php echo $form->textField($model,'descripcion',array('size'=>60,'maxlength'=>128)); ?>
<?php echo $form->error($model,'descripcion'); ?>
	</div>

	<div class="row">
			</div>

	<div class="row">
			</div>


<label for="UbicacionTec">Ubicación Técnica</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'padre0',
							'fields' => 'codigoSAP',
							'allowEmpty' => true,
							'style' => 'dropdownlist',
							)
						); ?>
			<label for="Usuarios">Usuarios que pertenecen a esta Ubicación Técnica</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'supervisor0',
							'fields' => 'Username',
							'allowEmpty' => false,
							'style' => 'dropdownlist',
							)
						); ?>
			
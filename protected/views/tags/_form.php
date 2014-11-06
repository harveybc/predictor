<p class="note">Campos con<span class="required">*</span> son necesarios.</p>

<?php echo $form->errorSummary($model); ?>

	<div class="row">
		<?php echo $form->labelEx($model,'descripcion'); ?>
<?php echo $form->textField($model,'descripcion',array('size'=>60,'maxlength'=>128)); ?>
<?php echo $form->error($model,'descripcion'); ?>
	</div>

	<div class="row">
			</div>


<label for="Documentos">Belonging Documentos</label><?php 
					$this->widget('application.components.Relation', array(
							'model' => $model,
							'relation' => 'documento0',
							'fields' => 'descripcion',
							'allowEmpty' => false,
							'style' => 'dropdownlist',
							)
						); ?>
			